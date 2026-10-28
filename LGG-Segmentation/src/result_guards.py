"""Validate saved results and preserve incorrectly placed prediction files.

This module deliberately does not import core or change any checkpoint.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

import pandas as pd


SUFFIXES = ('image', 'true', 'pred')


def expected_stems(frame, split):
    part = frame[frame['split'] == split]
    return {f'{row.patient}_{int(row.slice_id)}' for row in part.itertuples()}


def _record_moves(run, moves):
    if not moves:
        return
    path = run / 'prediction_cache_moves.json'
    existing = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    existing.extend(moves)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(existing, indent=2), encoding='utf-8')
    os.replace(temp, path)


def _move_preserving(run, source, destination, reason):
    if destination.exists():
        # Keep both copies when a prior validation export already exists.
        destination = run / 'cache_quarantine' / 'migration_conflicts' / source.name
        counter = 1
        while destination.exists():
            destination = destination.with_name(f'{source.stem}.{counter}{source.suffix}')
            counter += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                  source=source.relative_to(run).as_posix(),
                  destination=destination.relative_to(run).as_posix(),
                  sha256=digest, reason=reason)
    # Persist the intended move before touching historical evidence.
    _record_moves(run, [record])
    shutil.move(str(source), str(destination))
    return record


def migrate_validation_predictions(frame, run):
    """Move old validation PNGs out of the final-test directory, with an audit."""
    run = Path(run)
    source = run / 'predictions'
    moves = []
    for stem in sorted(expected_stems(frame, 'val')):
        for suffix in SUFFIXES:
            path = source / f'{stem}_{suffix}.png'
            if path.is_file():
                moves.append(_move_preserving(run, path,
                    run / 'validation_predictions' / path.name,
                    'Validation preview file found in final-test prediction directory'))
    if moves:
        print(f'{run.name}: preserved {len(moves)} misplaced validation PNGs; see prediction_cache_moves.json', flush=True)
    return moves


def isolate_prediction_cache(frame, run):
    """Preserve all non-test files before a test export; never delete evidence."""
    run = Path(run)
    moves = migrate_validation_predictions(frame, run)
    allowed = {f'{stem}_{suffix}.png' for stem in expected_stems(frame, 'test') for suffix in SUFFIXES}
    directory = run / 'predictions'
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name not in allowed:
                moves.append(_move_preserving(run, path,
                    run / 'cache_quarantine' / 'non_test_files' / path.name,
                    'File does not belong to the current held-out test split'))
    return moves


def training_complete(run, expected_signature, epochs):
    """Require a matching completion marker and all completed history epochs."""
    run = Path(run)
    try:
        marker = json.loads((run / 'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
        meta = json.loads((run / 'experiment.json').read_text(encoding='utf-8'))
        history = pd.read_csv(run / 'history.csv')
        best = json.loads((run / 'best_epoch.json').read_text(encoding='utf-8'))
        return (marker.get('signature') == expected_signature == meta.get('signature')
                and int(marker['epochs']) == int(epochs)
                and sorted(history['epoch'].tolist()) == list(range(1, int(epochs) + 1))
                and 1 <= int(best['epoch']) <= int(epochs)
                and (run / 'best.pt').is_file())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def test_cache_complete(config, frame, experiment, out, signature_value=None):
    """Check exact test identifiers, three-panel files, signatures and metrics."""
    run = Path(out) / 'experiments' / experiment['name']
    try:
        meta = json.loads((run / 'experiment.json').read_text(encoding='utf-8'))
        sig = signature_value if signature_value is not None else meta['signature']
        if not training_complete(run, sig, config['epochs']):
            return False
        marker = json.loads((run / 'EVALUATION_COMPLETE.json').read_text(encoding='utf-8'))
        if marker.get('signature') != sig or marker.get('best_weights_sha256') != hashlib.sha256((run/'best.pt').read_bytes()).hexdigest():
            return False
        part = frame[frame['split'] == 'test']
        expected = expected_stems(frame, 'test')
        if not expected:
            return False
        required = {f'{stem}_{suffix}.png' for stem in expected for suffix in SUFFIXES}
        actual = {p.name for p in (run / 'predictions').glob('*') if p.is_file()}
        if actual != required:
            return False
        slices = pd.read_csv(run / 'test_slices.csv')
        keys = list(zip(slices['patient'], slices['slice_id'].astype(int)))
        wanted = set(zip(part['patient'], part['slice_id'].astype(int)))
        patients = pd.read_csv(run / 'test_patients.csv')
        metrics = json.loads((run / 'test_metrics.json').read_text(encoding='utf-8'))
        return (len(keys) == len(wanted) and set(keys) == wanted
                and not patients['patient'].duplicated().any()
                and set(patients['patient']) == set(part['patient'])
                and int(metrics['slices']) == len(wanted)
                and int(metrics['patients']) == part['patient'].nunique())
    except (OSError, ValueError, KeyError, TypeError):
        return False

# CA6001 Group Assignment - Group 23

## Comparing U-Net and a No-Skip Encoder-Decoder for Brain MRI Segmentation

**This repository is the code submission for the CA6001 group assignment, prepared collaboratively by Group 23.**

- **Course:** CA6001 - AI Algorithms Fundamentals & Application
- **Group members:** LIU XUANHUI, LIU YAODONG, LIU YUHAN, YU HAILANLAN
- **Task:** binary segmentation of FLAIR abnormalities in brain MRI
- **Submission scope:** source code, review notebooks, and compact saved experiment results. The final report, presentation slides, and presentation recording are submitted separately.

## Project overview

We compare a 2D U-Net with an encoder-decoder that removes its skip connections (`no_skip`). Both use the same patient split, loss, learning rate, and five-epoch training budget. We evaluate patient-level overlap, positive-slice segmentation, and false positives on empty masks.

This is a course experiment. Its results describe FLAIR abnormality segmentation, not tumour diagnosis, grading, or survival prediction.

## Repository structure

| File or directory | Purpose |
| --- | --- |
| [src/core.py](src/core.py) | MRI loading, preprocessing, both models, loss, training, and evaluation |
| [src/workflow.py](src/workflow.py) | Experiment settings, patient split preparation, and workflow helpers |
| [src/charts.py](src/charts.py) | Metric aggregation, paired comparisons, and figure generation |
| [src/result_guards.py](src/result_guards.py) | Checks for completed training and cached predictions |
| [run_project.py](run_project.py) | Command-line entry point for a new experiment |
| [Main.ipynb](Main.ipynb) | Overview of the dataset and saved five-epoch results |
| [Analysis.ipynb](Analysis.ipynb) | Patient comparisons, empty-mask errors, and area-error analysis |
| [results/settings.json](results/settings.json) | Settings for the submitted experiment |
| [results/experiments/](results/experiments/) | Per-model training history, selected epoch, metrics, and architecture text |
| [results/analysis/tables/](results/analysis/tables/) | Saved patient/slice metrics and analysis tables |
| [requirements-analysis.txt](requirements-analysis.txt) | Dependencies for reviewing the saved results in notebooks |
| [requirements.txt](requirements.txt) | Dependencies for training and the full analysis pipeline |

Start with `Main.ipynb` for the results overview or `src/core.py` for the model implementation. The source and result tables can be inspected directly on GitHub.

## Dataset and split

Data source: [Kaggle LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation), published by Mateusz Buda. Our saved dataset summary contains **110 patients and 3,929 image-mask pairs**. Each input has three MRI channels; the target is one binary mask.

| Split | Patients | Slices | Positive masks | Empty masks |
| --- | ---: | ---: | ---: | ---: |
| Training | 78 | 2,753 | 962 | 1,791 |
| Validation | 16 | 627 | 230 | 397 |
| Test | 16 | 549 | 181 | 368 |

The split is by patient, so slices from the same patient stay in one partition. Images use per-slice, per-channel 1st/99th percentile normalization. Masks use nearest-neighbour resizing. See [dataset_summary.csv](results/analysis/tables/dataset_summary.csv).

## Models and training settings

Both networks have four encoder/decoder stages, a base width of 32, and a one-channel logits output. U-Net concatenates encoder features at matching decoder resolutions; `no_skip` removes these concatenations and consequently has fewer parameters.

| Setting | Value |
| --- | --- |
| Input size | 3 x 256 x 256 |
| Seed | 42 |
| Batch size | 4 |
| Optimizer / learning rate | Adam / 0.0001 |
| Loss | 0.5 BCEWithLogits + 0.5 soft Dice loss |
| Prediction threshold | 0.5 after sigmoid |
| Training budget | 5 cumulative epochs per model |
| Model selection | Highest validation patient-macro Dice |
| Data augmentation / LR scheduler | None / None |

**Five epochs means one supplied epoch followed by four continued epochs for each model; it is not five independent runs.** The first epoch used CPU BF16; continuation used CUDA FP16 AMP. Session metadata is retained in each model's `experiment.json`. The two `history.csv` files retain epochs 1 through 5.

## Saved results

The following values come from the retained tables, not from a new training run during packaging.

| Model | Patient-macro Dice | Patient-macro IoU | Selected epoch | Parameters | Empty-mask FP slices |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | 0.6716 | 0.5387 | 5 | 7,763,041 | 139 / 368 (37.77%) |
| No-skip | 0.6904 | 0.5635 | 3 | 6,979,681 | 24 / 368 (6.52%) |

Patient-level metrics pool pixel counts across each patient's slices before averaging equally across patients. An empty-mask false-positive slice is an empty ground-truth slice with at least one predicted foreground pixel.

U-Net has higher mean Dice on the 181 positive test slices (0.6820 versus 0.6276), while no-skip has fewer false-positive slices among the 368 empty test slices. The paired patient Dice difference, U-Net minus no-skip, is -0.0188, with a descriptive 95% bootstrap interval of [-0.0803, 0.0429]. This interval includes zero; this run does not establish a stable winner.

Evidence: [model comparison](results/analysis/tables/model_comparison.csv), [empty-mask errors](results/analysis/tables/empty_mask_errors.csv), [slice metrics](results/analysis/tables/all_slice_metrics.csv), and [paired uncertainty](results/analysis/tables/paired_difference_summary.json).

Limitations include one split and seed, 16 test patients, five epochs, unequal model capacity, and no external validation. An earlier 20-epoch branch accessed the same test set. The later five-epoch budget was user specified; the run record reports no tuning from test scores. This is therefore not a previously untouched external test.

## Optional execution

Reading the code and saved CSV/JSON tables does not require installing dependencies.

### Review the saved results

In a Python environment (the existing checks used Python 3.12), run these commands from the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python -m jupyterlab
```

Open `Main.ipynb` or `Analysis.ipynb`. Their default cells read the included tables and require no raw MRI data, trained weights, GPU, or PyTorch. Notebook outputs have been cleared to keep the repository small.

### Train a new experiment

Download and extract the dataset from the link above. Install a compatible PyTorch build using the [official installation guide](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python run_project.py --data-root "/path/to/kaggle_3m" --run-name trial_01 --epochs 5
```

Replace `/path/to/kaggle_3m` with your actual dataset directory. Both models will be trained and evaluated; new outputs are written to `runs/trial_01/`. The command does not overwrite the submitted `results/`. Dependencies specify minimum versions rather than a locked environment. A fresh run on another environment is not guaranteed to reproduce the archived numbers exactly, particularly because the archived run continued across CPU and GPU sessions.

The lightweight submission omits raw MRI images, trained checkpoints, and bulk prediction figures. Recreating the original saved prediction images or running inference with the original trained models requires the full backup and its checkpoints.

## References and acknowledgement

- Ronneberger, Fischer and Brox (2015), [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
- Mateusz Buda, [LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
- Mateusz Buda, [brain-segmentation-pytorch](https://github.com/mateuszbuda/brain-segmentation-pytorch), an architecture and task reference. This project uses its own supplied training pipeline and does not load that repository's pretrained weights.

AI-assisted tools supported code refinement and parts of the analysis implementation. The group is responsible for the submitted work and its interpretation.

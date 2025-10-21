# SleepFM (Sleep Foundation Model)

## 🔥 News
- SleepFM was accepted at ICML 2024!
- Shorter version of our paper acceped to ICLR TS4H workshop and AAAI 2024 SSS on Clinical FMs
- [Our paper](https://arxiv.org/abs/2405.17766v1) is out on arxiv.

## 📖 Introduction
Sleep is a complex physiological process evaluated through various modalities recording electrical brain, cardiac, and respiratory activities. We curate a large polysomnography dataset from over 14,000 participants comprising over 100,000 hours of multi-modal sleep recordings. Leveraging this extensive dataset, we developed SleepFM, the first multi-modal foundation model for sleep analysis. We show that a novel leave-one-out approach for contrastive learning significantly improves downstream task performance compared to representations from standard pairwise contrastive learning. A logistic regression model trained on SleepFM's learned embeddings outperforms an end-to-end trained convolutional neural network (CNN) on sleep stage classification (macro AUROC 0.88 vs 0.72 and macro AUPRC 0.72 vs 0.48) and sleep disordered breathing detection (AUROC 0.85 vs 0.69 and AUPRC 0.77 vs 0.61). Notably, the learned embeddings achieve 48% top-1 average accuracy in retrieving the corresponding recording clips of other modalities from 90,000 candidates. This work demonstrates the value of holistic multi-modal sleep modeling to fully capture the richness of sleep recordings.

# 📖 Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Licence](#license)

<a name="installation"/>

# 💿 Installation

Please use the following steps to create an environment for running SleepFM

```bash
git clone https://github.com/rthapa84/sleepfm-codebase.git
cd sleepfm-codebase
conda env create -f environment.yml
conda activate sleepfm_env
```

<a name="usage"/>

# 👩‍💻 Usage

*This repository now ships a complete CAP (Cyclic Alternating Pattern) workflow that reuses the released SleepFM checkpoint, prepares the dataset, trains a downstream classifier, and exports attribution maps for explainability.*

All executable scripts live in the `sleepfm/` directory and are numbered in the order they are typically invoked. Helper modules (`sleepfm/utils.py`, `sleepfm/config.py`) expose common configuration and plotting utilities.

## 1. Configure paths and environment

`sleepfm/config.py` defaults to a CAP-friendly layout. You can override the raw and processed directories as well as channel definitions with environment variables:

```bash
export SLEEPFM_CAP_RAW_DIR="/path/to/cap/raw"
export SLEEPFM_CAP_PROCESSED_DIR="/path/to/cap/processed"
```

Optional overrides are documented inline in `sleepfm/config.py` (e.g., `SLEEPFM_CAP_CHANNELS`, `SLEEPFM_CAP_ADDITIONAL_LABEL_MAPS`). The provided Conda environment already includes `mne`, `scikit-learn`, and PyTorch with CUDA support.

## 2. Preprocess CAP EDF files

`sleepfm/0_extract_pretraining_data.py` reads CAP PSG recordings (``*-PSG.edf``) and the accompanying hypnogram files, slices them into 30 second epochs, and stores the signals plus per-epoch metadata. Subject-level labels (e.g., disorder presence) can be supplied via CSV.

```bash
python sleepfm/0_extract_pretraining_data.py \
  --data_path /path/to/cap/raw \
  --save_path /path/to/cap/processed \
  --metadata_csv /path/to/metadata.csv \
  --metadata_id_column record_id \
  --metadata_label_column disorder \
  --label_key disorder \
  --chunk_duration 30 --target_sampling_rate 256 --num_threads 8
```

Each epoch is saved under `processed/X/<record>/<record>_<epoch>.npy` and the corresponding label dictionary (sleep stage, subject id, and optional disorder label) lives in `processed/Y/<record>.pickle`.

## 3. Create dataset splits

`sleepfm/1_prepare_dataset.py` builds subject-level splits and produces two pickle files:

* `dataset.pickle` keeps the hierarchical subject → label → event structure for contrastive pretraining.
* `dataset_events_*.pickle` flattens events into `(path, label)` pairs for downstream classifiers. Use `--label_key disorder` to request disorder labels instead of sleep stages.

```bash
python sleepfm/1_prepare_dataset.py \
  --dataset_dir /path/to/cap/processed \
  --label_key disorder \
  --num_threads 8
```

## 4. Stage the released SleepFM checkpoint

Copy `sleepfm/checkpoint/best.pt` into a run directory inside your processed dataset (for example `/path/to/cap/processed/cap_run/best.pt`). All downstream scripts expect the checkpoint next to their outputs.

## 5. Generate SleepFM embeddings

`sleepfm/3_generate_embed_pretraining.py` loads modality-specific encoders from the checkpoint and exports L2-normalised embeddings for each split. Pass the dataset event file that matches the labels you intend to train on:

```bash
python sleepfm/3_generate_embed_pretraining.py cap_run \
  --dataset_dir /path/to/cap/processed \
  --dataset_file dataset_events_disorder_-1.pickle \
  --splits train,valid,test
```

Embeddings are written to `/path/to/cap/processed/cap_run/eval_data/` and preserve modality ordering (respiratory, sleep stages, ECG).

## 6. Train an embedding-based classifier

`sleepfm/4_classification_eval_pretraining.py` now supports both scikit-learn and PyTorch backends. The default (`--trainer torch`) fits a linear head in PyTorch so gradients can flow back into the encoders for XAI. Results are saved under `models/`, `probs/`, and `figures/`.

```bash
python sleepfm/4_classification_eval_pretraining.py \
  --output_file cap_run \
  --dataset_dir /path/to/cap/processed \
  --dataset_event_file dataset_events_disorder_-1.pickle \
  --label_key disorder \
  --modality_type sleep_stages \
  --trainer torch \
  --max_epochs 100 --patience 15
```

The resulting checkpoint (e.g., `models/sleep_stages_disorder_linear.pt`) contains the linear weights, label mapping, and metadata required for attribution.

## 7. Generate attribution maps

Use `sleepfm/xai/generate_attributions.py` to compute Saliency, Integrated Gradients, SmoothGrad, and 1D Grad-CAM for any event. The script reuses the SleepFM encoder, the trained linear head, and the new attribution utilities under `sleepfm/xai/`.

```bash
python sleepfm/xai/generate_attributions.py \
  --dataset_dir /path/to/cap/processed \
  --output_file cap_run \
  --linear_checkpoint /path/to/cap/processed/cap_run/models/sleep_stages_disorder_linear.pt \
  --modality_type sleep_stages \
  --label_key disorder \
  --dataset_event_file dataset_events_disorder_-1.pickle \
  --split test --index 0 --methods saliency,integrated_gradients,gradcam
```

Attribution arrays (saved as `.npy`) and JSON metadata are emitted to `cap_run/xai_outputs/` by default.

## 8. End-to-end automation

The helper script `sleepfm/bash_scripts/cap_pipeline.sh` stitches the full workflow together. Set `RAW_DIR`, `DATASET_DIR`, and (optionally) `LABEL_KEY` before running the script to process the dataset, generate embeddings, and fit the downstream classifier in one go.

## BibTeX

```bibtex
@inproceedings{thapa2024sleepfm,
  title={SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals},
  author={Rahul Thapa and Bryan He and Magnus Ruud Kjaer and Hyatt Moore and Gauri Ganjoo and Emmanuel Mignot and James Zou},
  booktitle={International Conference on Machine Learning},
  year={2024},
}
```

## 🪪 License

[MIT License](LICENSE)

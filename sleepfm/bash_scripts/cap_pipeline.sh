#!/bin/bash

set -euo pipefail

source activate sleepfm_env

DATASET_DIR="${DATASET_DIR:-$HOME/data/cap/processed}"
RAW_DIR="${RAW_DIR:-$HOME/data/cap/raw}"
OUTPUT_NAME="${OUTPUT_NAME:-cap_run}"
METADATA_CSV="${METADATA_CSV:-$DATASET_DIR/metadata.csv}"
LABEL_KEY="${LABEL_KEY:-disorder}"
MODALITY="${MODALITY:-sleep_stages}"
LINEAR_TRAINER="torch"

python ../0_extract_pretraining_data.py \
  --data_path "$RAW_DIR" \
  --save_path "$DATASET_DIR" \
  --chunk_duration 30 \
  --target_sampling_rate 256 \
  --num_threads 8 \
  --metadata_csv "$METADATA_CSV" \
  --metadata_id_column record_id \
  --metadata_label_column "$LABEL_KEY" \
  --label_key "$LABEL_KEY"

python ../1_prepare_dataset.py \
  --dataset_dir "$DATASET_DIR" \
  --num_threads 8 \
  --label_key "$LABEL_KEY"

mkdir -p "$DATASET_DIR/$OUTPUT_NAME"
cp ../checkpoint/best.pt "$DATASET_DIR/$OUTPUT_NAME/best.pt"

DATASET_EVENT_FILE="dataset_events_${LABEL_KEY}_-1.pickle"
if [ "$LABEL_KEY" = "sleep_stage" ]; then
  DATASET_EVENT_FILE="dataset_events_-1.pickle"
fi

python ../3_generate_embed_pretraining.py "$OUTPUT_NAME" \
  --dataset_dir "$DATASET_DIR" \
  --dataset_file "$DATASET_EVENT_FILE" \
  --splits train,valid,test

python ../4_classification_eval_pretraining.py \
  --output_file "$OUTPUT_NAME" \
  --dataset_dir "$DATASET_DIR" \
  --dataset_event_file "$DATASET_EVENT_FILE" \
  --label_key "$LABEL_KEY" \
  --modality_type "$MODALITY" \
  --trainer "$LINEAR_TRAINER" \
  --max_epochs 100 \
  --patience 15


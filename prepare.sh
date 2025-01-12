#!/bin/bash

DATA_DIR="data/finetune"

python scripts/prepare_data_meta.py --data-dir $DATA_DIR
python scripts/prepare_mel.py --data-dir $DATA_DIR
python scripts/prepare_rms.py --data-dir $DATA_DIR
python scripts/prepare_f0.py --data-dir $DATA_DIR
python scripts/prepare_cvec.py --data-dir $DATA_DIR
python scripts/prepare_whisper.py --data-dir $DATA_DIR

python scripts/combine_features.py --data-dir $DATA_DIR
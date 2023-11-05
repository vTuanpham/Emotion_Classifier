#!/bin/bash
python src/train.py --output_dir "src/checkpoints" \
                    --data_path "src/data/fer_2013/fer2013/fer2013.csv"  \
                    --train_batch_size 256 \
                    --val_batch_size 1000  \
                    --train_size 0.8 \
                    --num_worker 4 \
                    --seed 42 \
                    --num_train_epochs 100 \
                    --learning_rate 1e-3 \
                    --dropout 0.25 \
                    --optim_name AdamW

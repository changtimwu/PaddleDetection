#!/bin/sh
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_file=$1 --output_dir=$OUTPUTDIR --device=gpu --enable_action=True

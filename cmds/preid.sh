#!/bin/sh
if [ ! -d $1 ]; then
    echo "$1 is not a directory"
    exit 
fi
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml  --video_dir=$1 --output_dir=$OUTPUTDIR --device=gpu

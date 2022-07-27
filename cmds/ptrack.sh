#!/bin/sh
othopts="--do_entrance_counting  --draw_center_traj"
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_file=$1 --output_dir=$OUTPUTDIR --device=gpu $othopts --model_dir det=ppyoloe

#!/bin/bash

ckptrandomred="pretrained_networks/red_1.pth,\
pretrained_networks/red_2.pth,\
pretrained_networks/red_3.pth,\
pretrained_networks/red_4.pth"

ckptrandomgreen="pretrained_networks/green_1.pth,\
pretrained_networks/green_2.pth,\
pretrained_networks/green_3.pth,\
pretrained_networks/green_4.pth"

ckptrandomblue="pretrained_networks/blue_1.pth,\
pretrained_networks/blue_2.pth,\
pretrained_networks/blue_3.pth,\
pretrained_networks/blue_4.pth"


bs=2

outputdir="output_3d_sl/4tmx/"


python inference_3d.py --channel 0 --prop_dist 100 --prop_dist_offset 2 \
    --data_path 'data/scenes_splitlohmann' --generator_path "$ckptrandomred" \
    --batch_size $bs --output_dir "$outputdir" --use_fp16  #--use_fp16 True

python inference_3d.py --channel 1 --prop_dist 100 --prop_dist_offset 2 \
    --data_path 'data/scenes_splitlohmann' --generator_path "$ckptrandomgreen" \
    --batch_size $bs --output_dir "$outputdir" --use_fp16 #--use_fp16 True

python inference_3d.py --channel 2 --prop_dist 100 --prop_dist_offset 2 \
    --data_path 'data/scenes_splitlohmann' --generator_path "$ckptrandomblue" \
    --batch_size $bs --output_dir "$outputdir" --use_fp16  #--use_fp16 True

python merge_RGB.py "$outputdir"
#!/bin/bash


PYTHONPATH=$(pwd):$PYTHONPATH python tools/test_joint.py --pretrained model_zoo/joint.pth \
                                                     --data_dir ./data/COVID-CS \
                                                     --file_list test.txt \
                                                     --input_features 1 \
                                                     --savedir ./outputs/joint_use_features \
                                                     --width 512 \
                                                     --height 512

#!/usr/bin/env bash

python=/Users/baidu/anaconda3/bin/python
if test -f $python; then
    echo "use baidu"
else
python=/home/cai.507/anaconda3/envs/ampligraph/bin/python
#python=/home/cai.507/anaconda3/bin/python
echo 'use osu'
fi



for method in 'ComplEx' 'DistMult'
do
for data in 'random' #'fb15-237' 'wn18rr'
do
$python test.py --method $method --n_epoch 26 --data $data
done
done

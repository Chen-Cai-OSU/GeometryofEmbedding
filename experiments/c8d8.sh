#!/usr/bin/env bash

# script for sacred
python=~/anaconda3/envs/ampligraph/bin/python
#python=~/anaconda3/bin/python
data='single_fam_tree'
start=1
end=30
n_rel=7
model='ComplEx'
n_node=1000
noise=-1

for model in 'ComplEx' 'DistMul'
do
time $python play.py with verbose=False seed='41' noise_rel=$noise model=$model rels='0 125 250 375 500 625 750 875 f 125f 250f 375f 500f 625f 750f 875f' n_node=$n_node data='cycref'
time $python play.py with verbose=False seed='41' noise_rel=$noise model=$model rels='0 125 250 375 500 625 750 875 f 125f 250f 375f 500f 625f 750f 875f' n_node=$n_node data='cycref'
#time $python play.py with verbose=False seed='41' noise_rel=$noise model=$model rels='0 250 500 750 f 250f 500f 750f' n_node=$n_node data='cycref'
done
exit
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
for n_node in 1001 1000 997
do
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='100 200 300 400 500 600' n_node=$n_node data='single_fam_tree'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='10 20 30 40 50 60' n_node=$n_node data='single_fam_tree'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 5 6' n_node=$n_node data='single_fam_tree'
done
done
exit

done
exit

time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5' n_node=$n_node data='single_fam_tree' n_epoch=300
exit

for n_epoch in 300
do
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='100 200 300 400 500 600' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='10 20 30 40 50 60' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5 6' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch


time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5 6' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5 6 7 8 9' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5 6 7 8 9 10 11 12' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15' n_node=$n_node data='single_fam_tree' n_epoch=$n_epoch
done
exit

start=0
end=28800
for ((i=start; i<=end; i++))
do
time $python play.py with verbose=False seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5 6 7 8' n_node=$n_node data='single_fam_tree' hindex=$i
#   echo "i: $i"
done
exit

for model in 'DistMul' 'ComplEx'
do
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 f1 f2 f3 f4' n_node=$n_node data='cycref'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 -1 -2 -3 -4' n_node=$n_node data='single_fam_tree'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 -501 -502 -503 -504' n_node=$n_node data='single_fam_tree'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 501 502 503 504' n_node=$n_node data='single_fam_tree'
done
exit

# different groups
for model in 'DistMul' #'ComplEx'
do
for missing_ratio in 0.1 0.2
do
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='f 1 2 3 4 f1 f2 f3 f4' n_node=$n_node data='cycref' missing_ratio=$missing_ratio
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='f 2 3 4 f1 f2 f3 f4' n_node=$n_node data='cycref' missing_ratio=$missing_ratio
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='f 1 2 4 f1 f2 f3 f4' n_node=$n_node data='cycref' missing_ratio=$missing_ratio
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 f1 f2 f3 f4' n_node=$n_node data='cycref' missing_ratio=$missing_ratio
done
done
exit

for model in 'DistMul'
do
for missing_ratio in 0.2
do
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 4 8 16 32' n_node=$n_node missing_ratio=$missing_ratio
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 3 9 27 81 243' n_node=$n_node missing_ratio=$missing_ratio
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 4 16 64 256 1024' n_node=$n_node missing_ratio=$missing_ratio
done
done
exit


# regualrity of relations
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 5 6' n_node=$n_node

time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 4 8 16 32' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 3 9 27 81 243' n_node=$n_node

time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='5 7 11 13 17 19' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='11 13 17 41 53 59' n_node=$n_node

time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 3 5 7 9 11' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='2 4 6 8 10 12' n_node=$n_node
exit

# first element of geometric sequence
model='DistMul'
time $python play.py with verbose=False  seed='41' noise_rel=$noise model=$model rels='1 2 4 8 16 32' n_node=$n_node
time $python play.py with verbose=False  seed='41' noise_rel=$noise model=$model rels='2 4 8 16 32 64' n_node=$n_node
time $python play.py with verbose=False  seed='41' noise_rel=$noise model=$model rels='4 8 16 32 64 128' n_node=$n_node
time $python play.py with verbose=False  seed='41' noise_rel=$noise model=$model rels='8 16 32 64 128 256' n_node=$n_node
time $python play.py with verbose=False  seed='41' noise_rel=$noise model=$model rels='16 32 64 128 256 512' n_node=$n_node
exit

# test TransE
time $python play.py with verbose=True seed='41' noise_rel=$noise model='TransE' rels='1 2 3 4 5 6' n_node=$n_node
exit



# individual relation
for model in 'DistMul' # 'ComplEx'
do
#time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 100 200 300 400 500' n_node=$n_node
for missing_ratio in 0 0.1 0.2 0.3 0.4
do
#time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 100 101 200 300 400 500' n_node=$n_node missing_ratio=$missing_ratio
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 99 100 101 200 300 400 500' n_node=$n_node missing_ratio=$missing_ratio
done
done
exit



# pertubation of relations 2 good
for model in 'DistMul' 'ComplEx'
do
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='100 200 300 400 500' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='101 199 301 399 499' n_node=$n_node
done
exit


# pertubation of relations 1 not good
for model in 'DistMul' 'ComplEx'
do
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 99 100 199 200 299 300 399 400 499 500' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 98 100 199 200 297 300 398 400 497 500' n_node=$n_node
done
exit


# step of geometric sequence
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 4 8' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 4 8 16' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 4 8 16 32' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 4 8 16 32 64' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 4 8 16 32 64 128' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 4 8 16 32 64 128 256' n_node=$n_node


# step of arthemtic sequence
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 5' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='5 10 15 20 25' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='10 20 30 40 50' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='50 100 150 200 250' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='100 200 300 400 500' n_node=$n_node
exit

# noise ratio and generalization
for ratio in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 8 16 24 256 264 272 280 512 520 528 536 768 776 784 792' n_node=32 data='cycprod' missing_ratio=$ratio
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 64 128 192 256 320 384 448 512 576 640 704 768 832 896 960' n_node=1024 data='single_fam_tree' missing_ratio=$ratio
done
exit

# noise ratio and generalization
for ratio in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='10 20 30 40 50 60 70 80 90' n_node=$n_node missing_ratio=$ratio
#time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='100 200 300 400 500 600 700 800 900' n_node=$n_node missing_ratio=$ratio
done
exit

time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='200 300 400 500 600 700 800 900' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='100 200 400 500 600 700 800 900' n_node=$n_node
exit

# product groups C_n versus C_m * C_m
# two groups (0 8 16 24)*(0 8 16 24) versus (0 64 ... 64*15) both are very good
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 8 16 24 256 264 272 280 512 520 528 536 768 776 784 792' n_node=32 data='cycprod'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 64 128 192 256 320 384 448 512 576 640 704 768 832 896 960' n_node=1024 data='single_fam_tree'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 11 22 363 374 385 726 737 748' n_node=33 data='cycprod'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 121 242 363 484 605 726 847 968' n_node=1089 data='single_fam_tree'
exit
# [0 2 4 6]*[0 2 4 6]
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 2 4 6 64 66 68 70 128 130 132 134 192 194 196 198' n_node=32 data='cycprod'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 2 4 6 64 66 68 70 128 130 132 134 192 194 196 198' n_node=1024 data='single_fam_tree'
# [0 1 3 5]*[0 1 3 5]
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 1 3 5 32 33 35 37 96 97 99 101 160 161 163 165' n_node=32 data='cycprod'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 1 3 5 32 33 35 37 96 97 99 101 160 161 163 165'  n_node=1024 data='single_fam_tree'
exit


# different groups
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 125 250 375 500 625 750 875' n_node=$n_node data='single_fam_tree'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 f 250 500 750 f250 250f 500f' n_node=$n_node data='cycref'
exit

# D_n: cycref
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 -501 -502 -503 -504' n_node=$n_node data='single_fam_tree'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 501 502 503 504' n_node=$n_node data='single_fam_tree'

exit
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='f 1 2 3 4 f1 f2 f3 f4' n_node=$n_node data='cycref'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 f1 f2 f3 f4' n_node=$n_node data='cycref'
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 -1 -2 -3 -4' n_node=$n_node data='single_fam_tree'



exit




# random relations
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='10 20 30 40' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='13 28 33 40' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='13 28 33 40 47' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='13 28 33 40 47 58' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='13 28 33 40 47 58 62 73 87' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='13 28 33 40 47 58 62 73 87 92 101 111' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='13 28 33 40 47 58 62 73 87 92 101 111 123 131 144' n_node=$n_node
exit

#time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 3' n_node=$n_node
#time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 10' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 3' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 4' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 5' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 6' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 7' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 8' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 9' n_node=$n_node
time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='0 -1 1 2 10' n_node=$n_node


#$python _play.py with rels='0 1 2 3 4 5' n_node=1000
exit
#$python _play.py with n_node='100' rels='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30'
$python _play.py with rels='2 4 6 8 10' n_node=100
$python _play.py with rels='2 4 6 8 10 12' n_node=100
$python _play.py with rels='2 4 6 8 10 12 14' n_node=100
$python _play.py with rels='2 4 6 8 10 12 14 16' n_node=100
$python _play.py with rels='2 4 6 8 10 12 14 16 18' n_node=100
$python _play.py with rels='2 4 6 8 10 12 14 16 18 20' n_node=100
$python _play.py with rels='2 4 6 8 10 12 14 16 18 20 22' n_node=100
$python _play.py with rels='2 4 6 8 10 12 14 16 18 20 22 24' n_node=100
$python _play.py with rels='2 4 6 8 10 12 14 16 18 20 22 24 26' n_node=100
$python _play.py with rels='2 4 6 8 10 12 14 16 18 20 22 24 26 28' n_node=100
$python _play.py with rels='2 4 6 8 10 12 14 16 18 20 22 24 26 28 30' n_node=100
exit


$python _play.py with rels='1 2 3 4 5 6' n_node=100
$python _play.py with rels='1 2 3 4 5 6 7 8' n_node=100
$python _play.py with rels='1 2 3 4 5 6 7 8 9 10 11 12' n_node=100
$python _play.py with rels='1 2 3 4 5 6 7 8 9 10 11 12 13 14' n_node=100
$python _play.py with rels='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16' n_node=100
$python _play.py with rels='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18' n_node=100
$python _play.py with rels='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20' n_node=100
$python _play.py with rels='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22' n_node=100
$python _play.py with rels='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24' n_node=100
$python _play.py with rels='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28' n_node=100



exit
for n_rel in 8 9 10
do
for step in $(seq 1 $end)
do
$python sacred_.py --step $step --n_node 1000 --n_rel $n_rel --seed 41 --data 'cycprod'
$python sacred_.py --step $step --n_node 1000 --n_rel $n_rel --seed 41 --data 'cycref'
done
done
exit
for model in 'DistMul'
do

for n_node in 997 1000 #2000
do

for noise in -1 1 2 3 4 5 6 7 8 9
do

time $python play.py with verbose=True seed='41' noise_rel=$noise model=$model rels='1 2 3 4 5 6 7 8 9' n_node=$n_node

done
done
done
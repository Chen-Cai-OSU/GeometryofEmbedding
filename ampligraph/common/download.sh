#!/usr/bin/env bash

# https://stackoverflow.com/questions/9427553/how-to-download-a-file-from-server-using-ssh
server=cai.507@CSE-SCSE101549D.cse.ohio-state.edu
file=/home/cai.507/Documents/DeepLearning/AmpliGraph-master/save_models/ComplEx-*
localdir=/Users/baidu/Documents/KG/AmpliGraph-master/save_models
scp $server:$file $localdir


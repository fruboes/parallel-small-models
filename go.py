#! /usr/bin/env python

import os

base_template = "LD_LIBRARY_PATH=/mnt/home/tfruboes/cuDNN/cuda-7.5/lib64/:/usr/local/cuda-7.5/targets/x86_64-linux/lib:/usr/local/cuda-7.5/targets/x86_64-linux/lib64 python -u S2_fit.py {}"

t0 = range(1,20)
t1 = [10,20,50, 100, 200]
t2 = [500, 1000, 2000]
#todo =  t0 + t1+ t2
todo = t0

os.system("rm loggpu.txt; touch loggpu.txt")
#os.system("rm logcpu.txt; touch logcpu.txt")

for i in todo:
    os.system(base_template.format(i) + " | tee -a loggpu.txt")


#base_template = "CUDA_VISIBLE_DEVICES= " + base_template
#for i in todo:
#    os.system(base_template.format(i) + " | tee -a logcpu.txt")


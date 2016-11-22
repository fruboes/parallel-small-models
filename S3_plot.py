#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ns = []
times = []
with open("loggpu.txt") as in_:
    for line in in_:
        if "Training took" not in line:
            continue
        line = line.split()
        n = int(line[-1])
        ttime = float(line[-2])
        ns.append(n)
        times.append(ttime)
        #print n, ttime


plt.plot(ns, times)
plt.xlabel("Number of models")
plt.ylabel("Training time")
plt.title("Parallel model training - 4 GPUs used")
#plt.show()
plt.savefig("/mnt/home/tfruboes/times.png")



import os

src11 = "data/kinetics400/kinetics_val_list_hd320.txt"
src12 = "data/kinetics400/kinetics_train_list_hd320.txt"
src21 = "data/kinetics400/kinetics400_val_list_videos.txt"
src22 = "data/kinetics400/kinetics400_train_list_videos.txt"

with open(src11, "r") as f:
    samples11 = {line.split(" ")[0] for line in f}
with open(src12, "r") as f:
    samples12 = {line.split(" ")[0] for line in f}
with open(src21, "r") as f:
    samples21 = {line[line.find("/") + 1:line.find("/") + 12] for line in f}
with open(src22, "r") as f:
    samples22 = {line[line.find("/") + 1:line.find("/") + 12] for line in f}

samples1 = samples11 | samples12
samples2 = samples21 | samples22

print(len(samples1), len(samples2), len(samples1 & samples2))
print(len(samples11), len(samples21), len(samples11 & samples21))
print(len(samples12), len(samples22), len(samples12 & samples22))
print(len(samples1 - samples2), len(samples2 - samples1))

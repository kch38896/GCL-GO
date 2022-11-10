import pickle
import torch
import pandas as pd
from pathlib import Path
from config import get_config
import numpy as np

args = get_config()
diamond_file = args.diamond_dir
go_SS_embed = torch.load(args.go_SS_embdding)
go_id = go_SS_embed["go_id"]
annotations = {}
train_annotations = {}
with open(args.test_data_dir, 'rb') as f:
    data = pickle.load(f)
    data = pd.DataFrame(data)
    for idx in range(len(data)):
        annotations[data["proteins"].loc[idx]] = data["annotations"].loc[idx]

with open(args.train_data_dir, 'rb') as f:
    data = pickle.load(f)
    data = pd.DataFrame(data)
    for idx in range(len(data)):
        train_annotations[data["proteins"].loc[idx]] = data["annotations"].loc[idx]
diamond_preds = {}
mapping = {}
test_bits = {}
test_train = {}

with open(diamond_file, 'r') as f:
    for lines in f:
        line = lines.strip('\n').split()
        if line[0] in test_bits:
            test_bits[line[0]].append(float(line[2]))
            test_train[line[0]].append(line[1])
        else:
            test_bits[line[0]] = [float(line[2])]
            test_train[line[0]] = [line[1]]

preds_score = []
for s in annotations:
    probs = [0] * len(go_id)
    probs = np.array(probs, dtype=np.float32)
    if str(s) in test_bits:
        weights = np.array(test_bits[str(s)]) / np.sum(test_bits[str(s)])
        for j in range(len(test_train[str(s)])):
            temp = [0] * len(go_id)
            temp = np.array(temp, dtype=np.float32)
            # print(s, j, test_train[str(s)])
            if test_train[str(s)][j] in train_annotations.keys():
                label = train_annotations[test_train[str(s)][j]]
                for l in label:
                    if l not in go_id:
                        continue
                    temp[go_id.index(l)] = 1.0
                probs += weights[j] * temp
    preds_score.append(probs)
preds_score = np.array(preds_score)
preds = []
optimal_a = {'mfo': 0.4, 'bpo': 0.5, 'cco': 0.6}
ypred = np.load('Data/CAFA3/test/MFO/MFO_pred_epoch19.npy')

for i, j in zip(preds_score, ypred):
    if np.sum(i) != 0:
        preds.append(i * (1 - 0.4) + j * 0.4)
    else:
        preds.append(j)

np.save('Data/CAFA3/test/MFO/MFO_pred_combine_diamond_epoch19.npy', preds)

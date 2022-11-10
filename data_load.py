import pickle
import numpy as np
from pathlib import Path
import pandas as pd
from file_utils import read_go_id
import torch
from torch.utils.data import DataLoader, Dataset
from aminoacid import to_onehot


class CustomDataset(Dataset):
    def __init__(self, go_id, path):
        super(CustomDataset, self).__init__()
        self.idx_map = go_id
        self.data_file = path
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            self.data = pd.DataFrame(data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seq = self.data["sequence"].loc[idx]
        seq = to_onehot(seq)
        seq_embed = self.data["embedding"].loc[idx]
        ant = self.data["annotations"].loc[idx]
        cls = [0] * len(self.idx_map)
        for a in ant:
            a = a.strip()
            if a not in self.idx_map:
                continue
            cls[self.idx_map.index(a)] = 1
        cls = np.array(cls, dtype=np.float32)
        # print("seq",len(seq), len(seq[seq == 1]))
        # print("cls",len(cls), len(cls[cls == 1]))
        cls = np.transpose(cls)
        seq = np.transpose(seq)
        return seq, seq_embed, cls

    def __len__(self):
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            data = pd.DataFrame(data)
        return len(data)


def dataset_read(args, go_id):
    # data loader
    train_dir = Path(args.train_data_dir)
    val_dir = Path(args.val_data_dir)
    test_dir = Path(args.test_data_dir)

    tr_dataset = CustomDataset(go_id, train_dir)
    train_dataset = DataLoader(dataset=tr_dataset, batch_size=args.batch_size)

    v_dataset = CustomDataset(go_id, val_dir)
    val_dataset = DataLoader(dataset=v_dataset, batch_size=args.batch_size)

    ts_dataset = CustomDataset(go_id, test_dir)
    test_dataset = DataLoader(dataset=ts_dataset, batch_size=args.batch_size)

    sp_list = read_go_id(Path(args.go_SS_dir))

    return train_dataset, val_dataset, test_dataset, sp_list

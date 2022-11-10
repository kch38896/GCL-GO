import argparse
from pathlib import Path

path = Path("Data/CAFA3")
onto = 'MFO'

parser = argparse.ArgumentParser()

parser.add_argument("--train_data_dir", type=str, default=path / "train/{}/{}_train_data_train.pkl".format(onto, onto))
parser.add_argument("--val_data_dir", type=str, default=path / "train/{}/{}_train_data_val.pkl".format(onto, onto))
parser.add_argument("--test_data_dir", type=str, default=path / "test/{}/{}_test_data.pkl".format(onto, onto))
parser.add_argument("--go_HR_dir", type=str, default=path / "train/{}/{}_HR_edge.tsv".format(onto, onto))
parser.add_argument("--go_SS_dir", type=str, default=path / "train/{}/{}_SS_edge.tsv".format(onto, onto))
parser.add_argument("--go_SS_embdding", type=str, default=path / "train/{}/{}_BioBERT.pth".format(onto, onto))
parser.add_argument("--diamond_dir", type=str, default=path / "test/diamond.tsv".format(onto))

parser.add_argument("--save_model_dir", type=str, default="Weights/{}".format(onto))
parser.add_argument("--checkpoint_dir", type=str, default="Checkpoint/{}".format(onto))
parser.add_argument('--true_file_dir', type=str, default=path / "test/{}/{}_true.npy".format(onto, onto))
parser.add_argument('--pred_file_dir', type=str, default=path / 'test/{}/{}_pred.npy'.format(onto, onto))
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--resume_from', type=bool, default=False)

parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run')
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument('--weight_decay', type=float, default=3e-4)

parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=32, help='mini-batch size')
parser.add_argument("--nhid", type=list, default=[512, 128])
parser.add_argument("--kernel_size", type=list,
                    default=[8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128])
parser.add_argument("--nneg", type=int, default=1, help='number of negative suffle sample')
parser.add_argument("--seq_embed_size", type=int, default=1024) # 1280 : esm-1b embeeding size / 1024 : seqvec embedding size
parser.add_argument("--print_every", type=int, default=5)
parser.add_argument("--maxlen", type=int, default=1024, help="maxlen of protein sequence")


def get_config():
    args = parser.parse_args()
    return args

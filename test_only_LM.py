import gc
import torch
import numpy as np
from pathlib import Path
from config import get_config
from data_load import dataset_read
from graph_utils import load_data, make_neighbor_graph


def main():
    args = get_config()

    gc.collect()
    torch.cuda.empty_cache()

    # Setting experimental environments
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # print("Before GPU Memory: %s" % getMemoryUsage(device))

    # Setting dataset
    go_SS_embed = torch.load(args.go_SS_embdding)
    go_id, go_embedding = go_SS_embed["go_id"], go_SS_embed["embedding"]
    go_embedding = go_embedding.to(device)
    train_data, test_data, sp_list = dataset_read(args, go_id)
    print("num_go_term:", len(go_id))
    adj, node_feat = load_data(Path(args.go_HR_dir), go_id, device)

    for j in range(4, 21):
        model = torch.load(Path('Weights/MFO') / 'weights_epoch_{}.pth'.format(j)).to(device)
        savepred = 'Data/CAFA3/test/MFO/MFO_pred_epoch{}.npy'.format(j)
        print("finished dataset loading")

        # Setting model
        model.eval()
        with torch.no_grad():
            tmp = 0
            for i, (seq_onehot, seq_embed, target) in enumerate(test_data):
                seq_onehot = seq_onehot.type(torch.FloatTensor).to(device)
                seq_embed = seq_embed.to(device)

                h_semantic, h_structure, pred = model(seq_embed, go_embedding, adj)

                if tmp == 0:
                    preds = pred
                    print("%d epoch" % (j), preds.size())
                    tmp = 1
                else:
                    preds = torch.cat([preds, pred])
                    print("%d epoch" % (j), preds.size())
            preds = preds.cpu().numpy()
            np.save(savepred, preds)

        print("finished model loading")


if __name__ == '__main__':
    main()

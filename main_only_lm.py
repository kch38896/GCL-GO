import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from config_bpo import get_config
from data_load import dataset_read
from graph_utils import load_data, make_neighbor_graph
from model_only_lm import Net

def main():
    args = get_config()

    gc.collect()
    torch.cuda.empty_cache()

    # Setting experimental environments
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    # Setting dataset
    go_SS_embed = torch.load(args.go_SS_embdding)
    go_id, go_embedding = go_SS_embed["go_id"], go_SS_embed["embedding"]
    train_data, val_data, test_data, sp_list = dataset_read(args, go_id)
    print("num_go_term:", len(go_id))
    adj, node_feat = load_data(Path(args.go_HR_dir), go_id, device)
    print("finished dataset loading")

    # Setting model
    model = Net(seq_feature=26, go_feature=1024, nhid=args.nhid, kernel_size=args.kernel_size, dropout=args.dropout).to(
        device)

    go_embedding = go_embedding.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    start = time.time()
    temp = start
    print("finished model loading")

    if args.train:
        total_loss = 0
        total_loss_mar_sem = 0
        total_loss_mar_str = 0
        total_loss_cl = 0
        for epoch in range(args.epochs):
            model.train()
            for i, (seq_onehot, seq_embed, target) in enumerate(train_data):
                seq_embed = seq_embed.to(device)
                target = target.type(torch.FloatTensor).to(device)
                optimizer.zero_grad()
                h_semantic, h_structure, pred = model(seq_embed, go_embedding, adj)
                h_semantic_p = make_neighbor_graph(h_semantic, go_id, sp_list, device)

                loss_mar_sem = 0
                loss_mar_str = 0
                for j in range(args.nneg):
                    indices = torch.randperm(h_semantic.size(0)).to(device)
                    h_semantic_n = torch.index_select(h_semantic, dim=0, index=indices).to(device)
                    loss_mar_sem += (triplet_loss(h_semantic, h_semantic_p, h_semantic_n)) / args.nneg
                    loss_mar_str += (triplet_loss(h_semantic, h_structure, h_semantic_n)) / args.nneg

                loss_cl = criterion(pred, target)
                loss = 0.5 * (loss_mar_sem + loss_mar_str) + 0.5 * loss_cl
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss_cl += loss_cl.item()
                total_loss_mar_sem += loss_mar_sem.item()
                total_loss_mar_str += loss_mar_str.item()

                ## record the average training loss, using something like
                if (i + 1) % args.print_every == 0:
                    avg_loss = total_loss / args.print_every
                    avg_loss_mar_sem = total_loss_mar_sem / args.print_every
                    avg_loss_mar_str = total_loss_mar_str / args.print_every
                    avg_loss_cl = total_loss_cl / args.print_every
                    print(
                        "time = %dm, epoch %d, iter = %d, loss = %.3f, loss_sem = %.3f, loss_str = %.3f, loss_cl = %.3f, %ds" % (
                            (time.time() - start) // 60,
                            epoch + 1, i + 1, avg_loss, avg_loss_mar_sem, avg_loss_mar_str, avg_loss_cl,
                            time.time() - temp))

                    total_loss = 0
                    total_loss_mar_sem = 0
                    total_loss_mar_str = 0
                    total_loss_cl = 0
                    temp = time.time()

            if (epoch + 1) % 1 == 0:
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'loss': loss
                }, '%s/epoch_%s.pth' % (args.checkpoint_dir, epoch + 1))
                torch.save(model, '%s/weights_epoch_%s.pth' % (args.save_model_dir, epoch + 1))

            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_loss_mar_sem = 0
                val_loss_mar_str = 0
                val_loss_cl = 0
                for i, (seq_onehot, seq_embed, target) in enumerate(val_data):
                    # print("onehot", seq_onehot.size())
                    seq_embed = seq_embed.to(device)
                    target = target.type(torch.FloatTensor).to(device)
                    h_semantic, h_structure, pred = model(seq_embed, go_embedding, adj)
                    h_semantic_p = make_neighbor_graph(h_semantic, go_id, sp_list, device)

                    for j in range(args.nneg):
                        indices = torch.randperm(h_semantic.size(0)).to(device)
                        h_semantic_n = torch.index_select(h_semantic, dim=0, index=indices).to(device)
                        loss_mar_sem += (triplet_loss(h_semantic, h_semantic_p, h_semantic_n)) / args.nneg
                        loss_mar_str += (triplet_loss(h_semantic, h_structure, h_semantic_n)) / args.nneg

                    loss_cl = criterion(pred, target)
                    loss = 0.5 * (loss_mar_sem + loss_mar_str) + 0.5 * loss_cl
                    # print("loss",loss)

                    val_loss += loss.item()
                    val_loss_cl += loss_cl.item()
                    val_loss_mar_sem += loss_mar_sem.item()
                    val_loss_mar_str += loss_mar_str.item()

                    ## record the average training loss, using something like
                    if (i + 1) % args.print_every == 0:
                        avg_loss = val_loss / args.print_every
                        avg_loss_mar_sem = val_loss_mar_sem / args.print_every
                        avg_loss_mar_str = val_loss_mar_str / args.print_every
                        avg_loss_cl = val_loss_cl / args.print_every
                        print(
                            "time = %dm, epoch %d, iter = %d, loss = %.3f, loss_sem = %.3f, loss_str = %.3f, loss_cl = %.3f, %ds" % (
                                (time.time() - start) // 60,
                                epoch + 1, i + 1, avg_loss, avg_loss_mar_sem, avg_loss_mar_str, avg_loss_cl,
                                time.time() - temp))

                        val_loss = 0
                        val_loss_mar_sem = 0
                        val_loss_mar_str = 0
                        val_loss_cl = 0
                        temp = time.time()


if __name__ == '__main__':
    main()

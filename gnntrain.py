import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from losses import AsymmetricLoss
from gnn_data import GNN_DATA
from gnn_model import GIN_Net2
from utils import Metrictor_PPI, print_file
from tensorboardX import SummaryWriter


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--description', default=None, type=str,
                    help='train description')
parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default=None, type=str,
                    help='protein sequence vector path')
parser.add_argument('--split_new', default=None, type=boolean_string,
                    help='split new index file or not')
parser.add_argument('--split_mode', default=None, type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--train_valid_index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--use_lr_scheduler', default=None, type=boolean_string,
                    help="train use learning rate scheduler or not")
parser.add_argument('--save_path', default=None, type=str,
                    help='model save path')
parser.add_argument('--graph_only_train', default=None, type=boolean_string,
                    help='train ppi graph conctruct by train or all(train with test)')
parser.add_argument('--batch_size', default=None, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--epochs', default=None, type=int,
                    help='train epoch number')
parser.add_argument('--device', default=None, type=str,
                    help='zzg')
parser.add_argument('--zzg', default=None, type=str,
                    help='zzg')

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=512, epochs=1000, scheduler=None,
          got=False):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0

    truth_edge_num = graph.edge_index.shape[1] // 2

    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)

        for step in range(steps):
            if step == steps - 1:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size:]
                else:
                    train_edge_id = graph.train_mask[step * batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size: step * batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            if got:
                output = model(graph.x, graph.edge_index_got, train_edge_id)
                label = graph.edge_attr_got[train_edge_id]
            else:
                output = model(graph.x, graph.edge_index, train_edge_id)
                label = graph.edge_attr_1[train_edge_id]

            label = label.type(torch.FloatTensor).to(device)

            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data, m(output).cpu().data)

            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1
            print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                       .format(epoch, step, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))

        valid_pre_result_list = []
        valid_label_list = []
        true_prob_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]

                output = model(graph.x, graph.edge_index, valid_edge_id)
                label = graph.edge_attr_1[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                loss = loss_fn(output, label)
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
                true_prob_list.append(m(output).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps

        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']),
                       save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, 'gnn_model_valid_best.ckpt'))

        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)

        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, AUPR:{}, Best valid_f1: {}, in {} epoch"
            .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1, metrics.Aupr,
                    global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)


def main():
    args = parser.parse_args()
    split_new = True
    split_mode = args.split_mode
    # split_mode = "dfs"

    #SHS27K
    # ppi_data = GNN_DATA(ppi_path="/opt/data/private/zzg/GNN_PPI-main/data/protein.actions.SHS27k.STRING.txt")
    # print("use_get_feature_origin")
    # ppi_data.get_feature_origin(pseq_path="/opt/data/private/zzg/GNN_PPI-main/data/protein.SHS27k.sequences.dictionary.tsv",
    #                             vec_path="/opt/data/private/zzg/zqgao22-HIGH-PPI-d5f6cba/protein_info/vec5_CTC.txt")

    #SHS148K
    # ppi_data = GNN_DATA(ppi_path="/opt/data/private/zzg/GNN_PPI-main/data/protein.actions.SHS148k.STRING.txt")
    # print("use_get_feature_origin")
    # ppi_data.get_feature_origin(pseq_path="/opt/data/private/zzg/GNN_PPI-main/data/protein.SHS148k.sequences.dictionary.tsv",
    #                             vec_path="/opt/data/private/zzg/GNN_PPI-main/data/vec5_CTC.txt")

    # #string
    ppi_data = GNN_DATA(ppi_path="/opt/data/private/zzg/GNN_PPI-main/data/9606.protein.actions.all_connected.txt")
    print("use_get_feature_origin")
    ppi_data.get_feature_origin(pseq_path="/opt/data/private/zzg/GNN_PPI-main/data/protein.STRING_all_connected.sequences.dictionary.tsv",
                                vec_path="/opt/data/private/zzg/GNN_PPI-main/data/vec5_CTC.txt")

    ppi_data.generate_data()

    print("----------------------- start split train and valid index -------------------")
    print("whether to split new train and valid index file, {}".format(split_new))
    if split_new:
        print("use {} method to split".format(split_mode))
    # ppi_data.split_dataset("D:/zzgppi/ppi/GNN_PPI-main/train_valid_index_json/random_string.json",
    #                        random_new=True, mode=args.split_mode)
    ppi_data.split_dataset("/opt/data/private/zzg/GNN_PPI-main/train_valid_index_json/bfs_string_test_1.json",
                           random_new=False, mode=split_mode)
    print("----------------------- Done split train and valid index -------------------")

    graph = ppi_data.data
    print(graph.x.shape)

    ppi_list = ppi_data.ppi_list

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    graph.edge_index_got = torch.cat(
        (graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    graph.edge_attr_got = torch.cat((graph.edge_attr_1[graph.train_mask], graph.edge_attr_1[graph.train_mask]), dim=0)
    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    device = args.device
    print(device)

    graph.to(device)

    model = GIN_Net2(in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1, pool_size=3, cnn_hidden=1, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    use_lr_scheduler = True

    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=True)

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    # loss_fn = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).to(device)

    save_path = "/opt/data/private/zzg/GNN_PPI-main/save_model"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    save_path = os.path.join(save_path, "gnn_{}".format(args.split_mode + '_mode_string_' + args.zzg, time_stamp))
    # save_path = os.path.join(save_path, "gnn_{}".format(split_mode + '_mode_string_gin', time_stamp))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    os.mkdir(save_path)
    with open(config_path, 'w') as f:
        args_dict = args.__dict__
        for key in args_dict:
            f.write("{} = {}".format(key, args_dict[key]))
            f.write('\n')
        f.write('\n')
        f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    summary_writer = SummaryWriter(save_path)

    train(model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=2048, epochs=320, scheduler=scheduler,
          got=False)

    summary_writer.close()


if __name__ == "__main__":
    main()
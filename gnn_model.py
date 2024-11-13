import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GATConv, GCNConv



class GIN_Net2(torch.nn.Module):
    def __init__(self, in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1, 
                hidden=512, use_jk=False, pool_size=3, cnn_hidden=1, train_eps=True, 
                feature_fusion=None, class_num=7, device=None):
        super(GIN_Net2, self).__init__()
        self.use_jk = use_jk
        self.train_eps = train_eps
        self.feature_fusion = feature_fusion
        # self.esm = pd.read_csv('/opt/data/private/zzg/GNN_PPI-main/data/esm2_shs27K.csv', header=None)
        # self.esm = pd.read_csv('/opt/data/private/zzg/GNN_PPI-main/data/esm2_shs148k.csv', header=None)
        # self.esm = pd.read_csv('/opt/data/private/zzg/zqgao22-HIGH-PPI-d5f6cba/protein_info/esm2_string.csv', header=None)

        self.conv1d = nn.Conv1d(in_channels=in_feature, out_channels=cnn_hidden, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
        self.lstm = nn.LSTM(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(in_len / pool_size), gin_in_feature)
        # self.fc = nn.Linear(1536, 256)
        self.fc = nn.Linear(256, 256)
        self.device = device

#GAT
        # self.conv1 = GATConv(256, 128, 4)
        # self.conv2 = GATConv(512, 128, 4)
        # self.conv3 = GATConv(512, 128, 4)
        # self.bn1_ = nn.BatchNorm1d(512)
        # self.bn2_ = nn.BatchNorm1d(512)
        # self.bn3_ = nn.BatchNorm1d(512)
        # self.fc1_ = nn.Linear(512, 512)
        # self.fc2_ = nn.Linear(512, 7)
#GCN
        # self.conv1 = GCNConv(256, 512)
        # self.conv2 = GCNConv(512, 512)
        # self.conv3 = GCNConv(512, 512)
        # self.bn1_ = nn.BatchNorm1d(512)
        # self.bn2_ = nn.BatchNorm1d(512)
        # self.bn3_ = nn.BatchNorm1d(512)
        # self.fc1_ = nn.Linear(512, 512)
        # self.fc2_ = nn.Linear(512, 7)

        self.gin_conv1 = GINConv( 
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.gin_convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
                    ), train_eps=self.train_eps
                )
            )
        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(num_layers*hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, class_num)
    
    def reset_parameters(self):
        
        self.conv1d.reset_parameters()
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()
        
        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.fc2.reset_parameters()
    
    def forward(self, x, edge_index, train_edge_id, p=0.5):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.maxpool1d(x)
        x = x.transpose(1, 2)
        x, _ = self.biGRU(x)
        # x, _ = self.lstm(x)
        x = self.global_avgpool1d(x)
        x = x.squeeze()
        x = self.fc1(x)

        # esm = np.array(self.esm)
        # esm = torch.tensor(esm)
        # esm = esm.to(torch.float32).to(self.device)
        # x = torch.cat([x, esm], dim=1)
        x = self.fc(x)


        # x = self.conv1(x, edge_index)
        # x = self.bn1_(x)
        # x = F.relu(x)  # 激活函数
        # x = self.conv2(x, edge_index)
        # x = self.bn2_(x)
        # x = F.relu(x)
        # x = self.conv3(x, edge_index)
        # x = self.bn3_(x)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=p, training=self.training)
        # x = self.lin2(x)

#GIN
        x = self.gin_conv1(x, edge_index)
        xs = [x]
        for conv in self.gin_convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.use_jk:
            x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)
        x = self.lin2(x)



        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        x = self.fc2(x)

        return x



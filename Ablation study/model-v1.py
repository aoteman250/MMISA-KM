
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp,global_mean_pool as gep,global_sort_pool,LayerNorm,TopKPooling
from cross_attention import *
from CBAM import *
from functools import reduce


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d = x.size()  # 获取输入张量的形状信息
        y = self.avg_pool(x)  # 应用自适应平均池化到输入张量
        y = y.view(b, c)  # 将输出视图重新形状以匹配 nn.Linear 的输入要求
        y = self.fc(y)  # 通过全连接层进行处理
        y = y.view(b, c, 1)  # 将输出视图重新形状以匹配输入张量的形状
        return x * y.expand_as(x)  # 将张量乘以得到的权重


class SKConv(nn.Module):
    def __init__(self, in_channels=320, out_channels=256, stride=1, M=2, r=8, L=32):
        '''
        :param in_channels: 输入通道维度
        :param out_channels: 输出通道维度，输入输出通道维度相同
        :param M: 分支数
        :param r: 特征Z的长度，计算其维度d时所需的比率
        :param L: 默认为32
        采用分组卷积： groups = 32，所以输入channel的数值必须是group的整数倍
        '''
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)  # 计算从向量C降维到向量Z的长度d
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()  # 根据分支数量添加不同核的卷积操作
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)  # 自适应池化到指定维度，这里指定为1，实现GAP
        self.fc1 = nn.Sequential(
            nn.Conv1d(out_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
        )  # 降维
        self.fc2 = nn.Conv1d(d, out_channels * M, kernel_size=1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1使得多个全连接层对应位置进行softmax,保证对应位置a+b+..=1

    def forward(self, input):
        input = input.transpose(1, 2)
        batch_size = input.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            output.append(conv(input))  # [batch_size, out_channels, d]

        # the part of fusion
        U = reduce(lambda x, y: x + y, output)  # 逐元素相加生成混合特征U [batch_size, channel, d]
        s = self.global_pool(U)  # [batch_size, channel, 1]
        z = self.fc1(s)  # S->Z降维 [batch_size, d, 1]
        a_b = self.fc2(z)  # Z->a，b升维 [batch_size, out_channels * M, 1]
        a_b = a_b.view(batch_size, self.M, self.out_channels, -1)  # 调整形状 [batch_size, M, out_channels, 1]
        a_b = self.softmax(a_b)  # 使得多个全连接层对应位置进行softmax [batch_size, M, out_channels, 1]

        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))  # 分割成多个 [batch_size, 1, out_channels, 1], [batch_size, 1, out_channels, 1]
        a_b = list(map(lambda x: x.view(batch_size, self.out_channels, 1), a_b))  # 调整形状 [batch_size, out_channels, 1]
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应不同卷积核输出的U逐元素相乘
        V = reduce(lambda x, y: x + y, V)  # 多个加权后的特征逐元素相加 [batch_size, out_channels, d]
        # V = self.max_pool(V).squeeze(2)
        # V = Reduce('b c t -> b c', 'max')(V)
        # V = V.transpose(1, 2)
        return V  # [batch_size, out_channels, d]






class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=256,  name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
            nn.ReLU(),
            # nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
            # nn.ReLU(),
        )
        self.name = name

    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return self.layers(x)


class ResDilaCNNBlocks(nn.Module):

    def __init__(self, feaSize, filterSize, blockNum=3, dilaSizeList=[1, 2, 4], dropout=0.2,
                 ):
        super(ResDilaCNNBlocks, self).__init__()  #
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize, filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}",
                                        ResDilaCNNBlock(dilaSizeList[i % len(dilaSizeList)], filterSize,
                                                        ))
        self.act = nn.ReLU()

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        # y = self.linear(x)  # => batchSize × seqLen × filterSize
        xt = x.transpose(1, 2)
        y = self.blockLayers(xt)  # => batchSize × seqLen × filterSize
        # y = self.act(y)  # => batchSize × seqLen × filterSize
        x = xt+y
        x = Reduce('b c t -> b c', 'max')(x)
        return x


class CNNNET(nn.Module):
    def __init__(self, input_channel=256):
        super(CNNNET, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.GELU())
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.GELU(),

)
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.GELU(),
            nn.Dropout(0.2),
)


        self.conv2_3 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.GELU(),
            nn.Dropout(0.2),
)

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel*3, out_channels=input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.GELU(),
            nn.Dropout(0.2),
        )




    def forward(self, x):
        x = self.conv1(x.transpose(1, 2))
        text1 = self.conv2_1(x)
        text2 = self.conv2_2(x)
        text3 = self.conv2_3(x)
        x = torch.cat([text1, text2, text3], dim=1)
        x = self.conv3(x)
        x = Reduce('b c t -> b c', 'max')(x)

        return x


class CNNLSTMModel(nn.Module):
    def __init__(self, window, dim, lstm_units, num_layers=2):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, lstm_units, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm1d(lstm_units),
            nn.LeakyReLU())


        self.SELayer = SELayer(512)
        self.SKConv = SKConv(in_channels=dim,out_channels=dim)


        self.cross = EncoderLayer(lstm_units , lstm_units , 0.2, 0.2, 4)  #交叉注意力机制

        self.ChannelGate = ChannelGate(gate_channels=128, reduction_ratio=12, pool_types=['avg', 'max'])
        self.SpatialGate = SpatialGate()

        self.maxPool = nn.MaxPool1d(window)
        self.drop = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(lstm_units, lstm_units, batch_first=True, num_layers=num_layers, bidirectional=False)
        self.act = nn.LeakyReLU()
        self.cls = nn.Linear(lstm_units , lstm_units)


    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in：bs, dim, window out: bs, lstm_units, window
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x = self.SKConv(x)
        return x



class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """

    def __init__(self, embed_size, head_num, dropout, residual=True):
        """
        """
        super(MultiHeadAttentionInteract, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num

        self.W_Q = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))
        self.W_K = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))
        self.W_V = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))
        self.act = nn.ReLU()

        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """

        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))

        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))

        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5

        attn_w = F.softmax(inner, dim=-1)

        attn_w = F.dropout(attn_w, p=self.dropout)

        results = torch.matmul(attn_w, Value)

        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)

        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))

        results = self.act(results)

        return results





class Selfattention(nn.Module):

    def __init__(self, field_dim, embed_size, head_num, dropout=0.2):
        super(Selfattention, self).__init__()



        self.vec_wise_net = MultiHeadAttentionInteract(embed_size=embed_size,
                                                       head_num=head_num,
                                                       dropout=dropout)

    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """

        b, f, e = x.shape
        vec_wise_x = self.vec_wise_net(x).reshape(b, f * e)
        # m_vec = self.trans_vec_nn(vec_wise_x)

        # m_x = m_vec
        return vec_wise_x
# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, embed_dim=256,n_output=1, num_features_pro=33, num_features_mol=78, dropout=0.4):
        super(GNNNet, self).__init__()

        self.embed_smile = nn.Embedding(65, embed_dim)
        self.embed_prot = nn.Embedding(26, embed_dim)
        self.onehot_smi_net = ResDilaCNNBlocks(embed_dim, embed_dim)
        self.onehot_prot_net = ResDilaCNNBlocks(embed_dim, embed_dim)
        self.CNN1 = CNNNET(input_channel=embed_dim)
        self.CNN2 = CNNNET(input_channel=embed_dim)
        # self.lstm1 = nn.LSTM(embed_dim, embed_dim, 4, batch_first=True,bidirectional=True)
        # self.lstm2 = nn.LSTM(embed_dim, embed_dim, 4, batch_first=True,bidirectional=True)
        self.CNNLSTM1 = CNNLSTMModel(window=100, dim=embed_dim, lstm_units=embed_dim, num_layers=4)
        self.CNNLSTM2 = CNNLSTMModel(window=500, dim=embed_dim, lstm_units=embed_dim, num_layers=4)

        self.mix_attention_layer = nn.MultiheadAttention(embed_dim*2, 4)
        self.linear = nn.Linear(480,128)
        self.mol_max_pool = nn.MaxPool1d(100)
        self.pro_max_pool = nn.MaxPool1d(500)

        self.down_sample = nn.Linear(256,128)



        self.smi_attention_poc = EncoderLayer(256, 256, 0.1, 0.1, 4)  #交叉注意力机制
        self.seq_attention_tdlig = EncoderLayer(256, 256, 0.1, 0.1, 4)


        print('GNNNet Loaded')
        self.n_output = n_output

        self.mol_conv1 = GATConv(num_features_mol, num_features_mol,heads=8)
        self.mol_conv2 = GCNConv(num_features_mol * 8, num_features_mol * 8)
        # self.mol_conv3 = GATConv(num_features_mol* 8, num_features_mol, heads=8,)

        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 8* 2, 512)
        self.mol_fc_g2 = torch.nn.Linear(512, 256)
        # self.mol_fc_g2 = torch.nn.Linear(256, output_dim)


        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GATConv(num_features_pro, num_features_pro,heads=8)
        self.pro_conv2 = GCNConv(num_features_pro * 8, num_features_pro * 8)
        # self.pro_conv3 = GATConv(num_features_pro* 8, num_features_pro, heads=8,)

        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro*8* 2, 512)
        self.pro_fc_g2 = torch.nn.Linear(512, 256)
        # self.pro_fc_g2 = torch.nn.Linear(1000, output_dim)


        self.ecfp1 = nn.Linear(1024, 512)
        self.ecfp2 = nn.Linear(512, 128)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        # self.fc1 = nn.Linear(2 * output_dim, 512)
        self.fc1 = nn.Linear(embed_dim*2, embed_dim*1)
        self.out = nn.Linear(embed_dim*1, self.n_output)


        proj_dim = embed_dim
        field_dim = 2
        self.feature_interact = Selfattention(field_dim=field_dim, embed_size=proj_dim, head_num=8)
        self.norm = nn.LayerNorm(embed_dim)

    def Elem_feature_Fusion_D(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        x_c = self.down_sample1(torch.cat((xs, x), dim=1))
        x_c = self.merge_atten(x_c)
        xs_ = self.leaky_relu(self.W1(xs))
        x_ = self.leaky_relu(self.W2(x))
        xs_m = x_c * xs_ + xs
        ones = torch.ones(x_c.shape)
        x_m = (ones - x_c) * x_ + x
        ys = xs_m + x_m
        return ys



    def forward(self, smile,sequence,data_mol,data_pro):
        # ecfp = ecfp.float()


        smile_vectors_onehot = self.embed_smile(smile)
        proteinFeature_onehot = self.embed_prot(sequence)

        mol_pair = self.CNNLSTM1(smile_vectors_onehot)
        pro_pair = self.CNNLSTM2(proteinFeature_onehot)



        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)
        # x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)



        x = torch.cat((gep(x, mol_batch),gmp(x, mol_batch)), 1)


        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.relu(self.mol_fc_g2(x))
        x = self.dropout(x)

        # x = torch.cat((x1,x2), 1)
        # x = self.mol_fc_g2(x)
        # x = self.dropout(x)
        #
        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)


        xt = torch.cat((gep(xt, target_batch),gmp(xt, target_batch)), 1)

        #
        # # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.relu(self.pro_fc_g2(xt))
        xt = self.dropout(xt)



        all_features = torch.stack([mol_pair,pro_pair], dim=2)
        all_features = self.norm(all_features.permute(0, 2, 1))
        all_features = self.feature_interact(all_features)

        xc = self.leaky_relu(self.fc1(all_features))
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

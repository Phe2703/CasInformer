import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch_geometric.nn import GCN2Conv, GATv2Conv, BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN

from layers_third.informer import (AttentionLayer, ProbAttention, FullAttention,
                            ProbMask, TriangularCausalMask,
                            Encoder, EncoderLayer, EncoderStack, ConvLayer,
                            Decoder, DecoderLayer,
                            )
# from layers_third.vaes import StaticFlowVAE


class GCN2(nn.Module):
    """
    借鉴GCN封装方式，GCN2的直接调用
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 3, dropout = 0. ,alpha= 0.1, theta= 0.5):
        super(GCN2, self).__init__()

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, i+1))

    def forward(self, node_features, edges):
        x = x_0 = self.fc1(node_features).relu()
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x,x_0, edges), inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        # x = F.relu(x, inplace=False)

        return x

class GATv2(BasicGNN):
    """
    借鉴GAT封装方式，GATv2的直接调用
    """
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        kwargs = copy.copy(kwargs)
        if 'heads' in kwargs and out_channels % kwargs['heads'] != 0:
            kwargs['heads'] = 1
        if 'concat' not in kwargs or kwargs['concat']:
            out_channels = out_channels // kwargs.get('heads', 1)

        return GATv2Conv(in_channels, out_channels, dropout=self.dropout,
                       **kwargs)

class DynamicRouting(nn.Module):
    '''
    动态路由层
    '''
    def __init__(self, in_dim, out_dim, num_iterations = 3):
        super(DynamicRouting,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_iterations = num_iterations
        self.W = nn.Parameter(torch.randn(1,in_dim,out_dim))  # 随机初始化网络参数
        # print(self.W)
    
    def forward(self,x):
        device = x.device
        num_nodes = x.size(1)  # 级联节点数
        batch_size = x.size(0)  # 快照数

        # W = torch.cat([self.W] * batch_size, dim=0).to(device) # 每个快照对应一个参数矩阵
        x = torch.matmul(x, self.W)

        # input_sum = torch.sum(input, dim=-1, keepdim=False)
        
        r_similarity = torch.zeros([batch_size, num_nodes], device=device)
        # b = Variable(b)

        label = torch.ones([batch_size,num_nodes], device=device)
        zero = torch.tensor(0., device=device)
        # label = torch.clone(r_sum)
        label = torch.where(torch.sum(x, dim=-1, keepdim=False) == 0, label, zero)

        r_similarity.masked_fill_(label.bool(), -float('inf')) 

        for i in range(self.num_iterations):

            weight_coeff = F.softmax(r_similarity, dim=-1).unsqueeze(dim=1)

            x_g = torch.matmul(weight_coeff, x)
            # x_g_all = torch.cat([x_g] * num_nodes, dim=1)

            r_similarity = F.cosine_similarity(x, x_g, dim=-1)
            r_similarity.masked_fill_(label.bool(), -float('inf'))
            # r_similarity = r_similarity

        return x_g.squeeze(dim=1)
       

class LSTMWrapper(nn.LSTM):
    '''
    input shape(batch_size, seq_length, input_size)
    lengths shape(batch_size)，每个句子的长度，无序
    '''
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, lengths):
        """
        https://blog.csdn.net/winter2121/article/details/115239554
        https://www.cnblogs.com/xiximayou/p/15036715.html
        """
        total_length = x.shape[self.batch_first]
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        x, _ = super().forward(x)
        x, _ = pad_packed_sequence(x, batch_first=self.batch_first, total_length=total_length)
        return x
        # return x[:,-1,:]

class GRUWrapper(nn.GRU):
    '''
    input shape(batch_size, seq_length, input_size)
    lengths shape(batch_size)，每个句子的长度，无序
    '''
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, lengths):
        """
        https://blog.csdn.net/winter2121/article/details/115239554
        https://www.cnblogs.com/xiximayou/p/15036715.html
        """
        total_length = x.shape[self.batch_first]
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        x, _ = super().forward(x)
        x, _ = pad_packed_sequence(x, batch_first=self.batch_first, total_length=total_length)
        return x

class TimeDecay(nn.Module):
    '''
    模拟时间衰减效应
    应该也可以作为位置编码使用
    '''
    def __init__(self, number_of_snapshot) -> None:
        super().__init__()
        self.number_of_snapshot = number_of_snapshot
        self.weight_decay = nn.Parameter(torch.randn((number_of_snapshot, number_of_snapshot),
                                        requires_grad=True))
        # print(self.weight_decay)
    
    def forward(self, batch_cascade, batch_time_interval):
        # (batch, number_of_snapshot) * (number_of_snapshot, number_of_snapshot)
        time_weight = torch.matmul(batch_time_interval, self.weight_decay)
        # (batch, number_of_snapshot)
        time_weight = time_weight.view(-1, 1)
        # (batch * number_of_snapshot)

        x_origin_size = tuple(batch_cascade.shape)  
        # (batch , number_of_snapshot, feature_dim)
        batch_cascade = batch_cascade.view(-1, x_origin_size[-1])
        # (batch * number_of_snapshot, feature_dim)

        batch_cascade = torch.mul(batch_cascade, time_weight)
        # (batch * number_of_snapshot, feature_dim)

        batch_cascade = batch_cascade.reshape(x_origin_size)
        # (batch , number_of_snapshot, feature_dim)

        # cascade_after_decay = cascade_after_decay.sum(1)
        # (batch , feature_dim)

        return batch_cascade

class TimeDecayWrapper(TimeDecay):
    '''
    TimeDecay的包装层
    padding 0 后输入时间衰减层并返回
    '''
    def __init__(self, number_of_snapshot) -> None:
        super().__init__(number_of_snapshot)
    
    def forward(self, batch_cascade, batch_time_interval, padding = True):
        # 输入x长度不定，可能需要padding
        # 输入y长度固定，无需再次padding
        current_device = batch_cascade.device

        if padding:
            # batch中最长的序列可能也比预设的snapshot的数量要少，因此需要扩充维度确保一致后再输入
            current_size = tuple(batch_time_interval.shape)
            padding_zero = torch.zeros( current_size[0],
                                        self.number_of_snapshot - current_size[1],
                                        device = current_device
                                    )
            batch_time_interval = torch.cat( (batch_time_interval, padding_zero), dim = 1)
            
            # x特征矩阵同理
            current_size = tuple(batch_cascade.shape)
            padding_zero = torch.zeros( current_size[0],
                                        self.number_of_snapshot - current_size[1],
                                        current_size[2],
                                        device = current_device
                                    )
            batch_cascade = torch.cat( (batch_cascade, padding_zero), dim = 1)

        # 输入到时间衰减层
        batch_cascade = super().forward(batch_cascade, batch_time_interval)

        return batch_cascade


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# class NFVAE(StaticFlowVAE):
#     '''
#     VAE + NF
#     目前没有找到合适的使用方式，负提升
#     '''
#     def __init__(self, in_dim, hidden_dim, latent_dim, layer=2, gate=0, flow='none', length=2):
#         super().__init__(in_dim, hidden_dim, latent_dim, layer, gate, flow, length)
    
#     def forward(self, x):
#         # x_in, log_det = logit_transform(x)
#         # x_hat, loss = super().forward(x_in)
#         # x_hat, loss = super().forward(x)

#         return super().forward(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)

        return self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)

class TokenEmbeddingWrapper(TokenEmbedding):
    def __init__(self, c_in, d_model, number_of_snapshot):
        super().__init__(c_in, d_model)
        self.number_of_snapshot = number_of_snapshot
    
    def forward(self, x, padding = False):
        
        if padding:
            current_device = x.device
            current_size = tuple(x.shape)
            padding_zero = torch.zeros( current_size[0],
                                        self.number_of_snapshot - current_size[1],
                                        current_size[2],
                                        device = current_device
                                    )
            x = torch.cat( (x, padding_zero), dim = 1)
        
        return super().forward(x)

class LinearEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, number_of_snapshot) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim,out_dim)
        self.number_of_snapshot = number_of_snapshot
    
    def forward(self, x, padding = False):
        if padding:
            current_device = x.device
            current_size = tuple(x.shape)
            padding_zero = torch.zeros( current_size[0],
                                        self.number_of_snapshot - current_size[1],
                                        current_size[2],
                                        device = current_device
                                    )
            x = torch.cat( (x, padding_zero), dim = 1)
        
        return self.linear(x)

class TimeFeatureEmbedding(nn.Module):
    def __init__(self,time_feature_dim, out_dim) -> None:
        super().__init__()
        self.embedding_layer = nn.Linear(time_feature_dim, out_dim)
    
    def forward(self, x):
        return self.embedding_layer(x)

class TimeFeatureEmbeddingWrapper(TimeFeatureEmbedding):
    def __init__(self, time_feature_dim, out_dim, number_of_snapshot) -> None:
        super().__init__(time_feature_dim, out_dim)
        self.number_of_snapshot = number_of_snapshot
        self.batch_norm = BatchNorm(number_of_snapshot)
    
    def forward(self, x, padding = False):
        
        if padding:
            current_device = x.device
            current_size = tuple(x.shape)
            padding_zero = torch.zeros( current_size[0],
                                        self.number_of_snapshot - current_size[1],
                                        current_size[2],
                                        device = current_device
                                    )
            x = torch.cat( (x, padding_zero), dim = 1)
        
        x = super().forward(x)
        x = self.batch_norm(x)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, number_of_snapshot, 
                time_feature_dim, cascade_feature_dim, out_dim, 
                embedding_layer_dropout) -> None:
        super().__init__()
        self.number_of_snapshot = number_of_snapshot
        self.pos_embedding_layer = TimeDecayWrapper(number_of_snapshot)
        self.sin_pos_embedding_layer = PositionalEmbedding(out_dim)
        self.token_embedding_layer = TokenEmbeddingWrapper(cascade_feature_dim, out_dim, number_of_snapshot)
        self.time_embedding_layer = TimeFeatureEmbeddingWrapper(time_feature_dim, out_dim, number_of_snapshot)
        self.drop_out = nn.Dropout(p = embedding_layer_dropout)
        self.batch_norm = BatchNorm(number_of_snapshot)
    
    def forward(self, x, batch_time_interval, batch_struct_time_feature, batch_cascade_feature, padding):
        # 只有输入的x需要考虑padding，y不存在这种情况

        x = self.pos_embedding_layer(x, batch_time_interval, padding) + \
            self.time_embedding_layer(batch_struct_time_feature, padding) + \
            self.token_embedding_layer(batch_cascade_feature, padding)
        
        x = self.drop_out(x)

        return x

class InformerEncoderWrapper(nn.Module):
    def __init__(self, hidden_dim, d_ff, factor = 5, attention_dropout = 0.0, n_heads = 8, 
                encoder_layer_num = 3, output_attention = False, attention_type = 'prob',
                activation='gelu', distil = True, mix = True,
                ) -> None:
        super().__init__()
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(ProbAttention(False, factor, attention_dropout = attention_dropout, output_attention=output_attention), 
                                hidden_dim, n_heads, mix = False),
                    hidden_dim,
                    d_ff,
                    dropout = attention_dropout,
                    activation=activation
                ) for l in range(encoder_layer_num)
            ],
            [
                ConvLayer(
                    hidden_dim
                ) for l in range(encoder_layer_num - 1)
            ] if distil else None,
            norm_layer = torch.nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        # enc_out, attns = self.encoder(x, attn_mask = ProbMask)

        return self.encoder(x, attn_mask = ProbMask)

class InformerDecoderWrapper(nn.Module):
    def __init__(self, hidden_dim, d_ff, factor = 5, attention_dropout = 0.0, n_heads = 8, 
                decoder_layer_num = 3, 
                activation='gelu', mix = True,
                ) -> None:
        super().__init__()
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(ProbAttention(True, factor, attention_dropout=attention_dropout, output_attention=False), 
                                hidden_dim, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=attention_dropout, output_attention=False), 
                                hidden_dim, n_heads, mix=False),
                    hidden_dim,
                    d_ff,
                    dropout=attention_dropout,
                    activation=activation,
                )
                for l in range(decoder_layer_num)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # out = self.decoder(x, cross, x_mask, cross_mask)

        return self.decoder(x, cross, x_mask, cross_mask)
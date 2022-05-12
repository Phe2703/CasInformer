import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCN, GAT, BatchNorm, LayerNorm, GraphNorm
from torch_sparse import SparseTensor

from layers import (GCN2, GATv2,
                DynamicRouting, 
                LSTMWrapper, GRUWrapper,
                # NFVAE,
                TokenEmbedding, TokenEmbeddingWrapper, LinearEmbedding,
                TimeDecayWrapper, TimeFeatureEmbeddingWrapper, 
                DataEmbedding,
                InformerEncoderWrapper, InformerDecoderWrapper
                )

train_logger = logging.getLogger('models')


class CasInformer(nn.Module):
    def __init__(self, model_settings, data_settings,  train_settings):
        """
        """
        super(CasInformer, self).__init__()
        self.model_settings = model_settings
        self.data_settings = data_settings
        self.train_settings = train_settings
        self.number_of_snapshot = self.data_settings['number_of_snapshot']
        self._setup_layers()

    def _setup_GNN_layer(self):
        gnn_dict = {
            "GCN":GCN,
            "GAT":GAT,
            "GCN2":GCN2,
            "GAT2":GATv2,
        }
        self.GNN_layer = gnn_dict[self.model_settings['gnn_type']](
            in_channels = self.data_settings['dim_of_node_features'],
            hidden_channels = self.model_settings['gnn_hidden_channels'],
            num_layers = self.model_settings['gnn_num_layers'],
            out_channels = self.model_settings['gnn_out_channels'],
            dropout=self.model_settings['gnn_dropout'],
        )
    
    def _setup_BN_layer(self):
        self.batch_norm_layer_1 = BatchNorm(self.model_settings['gnn_out_channels'])
        self.batch_norm_layer_2 = BatchNorm(self.model_settings['capsule_out_dim'])
        self.batch_norm_layer_3 = BatchNorm(self.model_settings['rnn_hidden_size'])
    
    def _setup_Aggregatting_layer(self):
        self.aggregate_layer = DynamicRouting(in_dim=self.model_settings['gnn_out_channels'], 
                                                out_dim=self.model_settings['capsule_out_dim'], 
                                    num_iterations=self.model_settings['dynamic_route_iter'])
    
    def _setup_RNN_layer(self):
        rnn_dict = {
            "LSTM":LSTMWrapper,
            "GRU":GRUWrapper,
        }
        if self.model_settings['rnn_type'] not in rnn_dict.keys():
            return None
        self.RNN_layer = rnn_dict[self.model_settings['rnn_type']](
                            input_size=self.model_settings['capsule_out_dim'] + (self.model_settings['cascade_level_features_dim'] if self.data_settings['use_cascade_level_features_in_rnn'] else 0),
                            hidden_size=self.model_settings['rnn_hidden_size'] // 2 if self.model_settings['rnn_bidirectional'] else self.model_settings['rnn_hidden_size'],
                            batch_first = True,
                            num_layers=self.model_settings['rnn_num_layers'],
                            dropout=self.model_settings['rnn_dropout'],
                            bidirectional=self.model_settings['rnn_bidirectional'],
                            )
        
        # for name, param in self.RNN_layer.named_parameters():
        #     if name.startswith("weight"):
        #         nn.init.kaiming_normal_(param)
        #     else:
        #         nn.init.zeros_(param)
    
    # def _setup_vae_layer(self):
    #     self.vae_layer = NFVAE(
    #         in_dim=self.model_settings['capsule_out_dim'],
    #         hidden_dim=self.model_settings['capsule_out_dim'] * 2,
    #         latent_dim=self.model_settings['capsule_out_dim'],
    #         gate=1,
    #         flow='radial',
    #         length=2,
    #     )
    
    def _setup_cascade_embedding_layer(self):
        self.cascade_embedding_layer = nn.Linear(
            in_features=self.data_settings['dim_of_cascade_features'],
            out_features=self.model_settings['cascade_level_features_dim'],
        )
        # self.cascade_embedding_layer = TokenEmbedding(
        #     c_in=self.data_settings['dim_of_cascade_features'],
        #     d_model=self.model_settings['cascade_level_features_dim'],
        # )
    
    def _setup_time_decay_layer(self):
        self.TimeDecay_layer = TimeDecayWrapper(self.number_of_snapshot)
    
    def _setup_output_layer(self):
        self.output_layer = MLP(
            in_channels=self.model_settings['rnn_hidden_size'],
            hidden_channels=self.model_settings['mlp_hidden_size'],
            out_channels=self.model_settings['mlp_outsize'], 
            num_layers=self.model_settings['mlp_num_layers'],
            dropout=self.model_settings['mlp_dropout'],
            batch_norm= False,
        )
        # for m in self.output_layer.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, a=1,mode='fan_in',nonlinearity='leaky_relu')
    
    def _setup_informer_layers(self):
        self.informer_encoder_layer = InformerEncoderWrapper(
            hidden_dim = self.model_settings['capsule_out_dim'],
            d_ff = self.model_settings['capsule_out_dim'],
            attention_dropout=self.model_settings['attention_dropout'],
        )
        self.informer_decoder_layer = InformerDecoderWrapper(
            hidden_dim = self.model_settings['capsule_out_dim'],
            d_ff = self.model_settings['capsule_out_dim'],
            attention_dropout=self.model_settings['attention_dropout'],
        )
        

        self.informer_embedding_layer = DataEmbedding(
            number_of_snapshot=self.number_of_snapshot,
            time_feature_dim=self.data_settings['time_tuple_features_dim'],
            out_dim=self.model_settings['capsule_out_dim'],
            embedding_layer_dropout = self.model_settings['attention_dropout'],
            cascade_feature_dim=self.data_settings['dim_of_cascade_features'],
        )
        self.informer_batch_norm_layer_1 = nn.BatchNorm1d(num_features=self.number_of_snapshot)
        self.informer_batch_norm_layer_2 = nn.BatchNorm1d(num_features=self.number_of_snapshot)
        self.informer_output_layer = nn.Linear(self.model_settings['capsule_out_dim'],
                                                self.model_settings['mlp_outsize'],
                                                )
        # for m in self.output_layer.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, a=1,mode='fan_in',nonlinearity='leaky_relu')

    def _setup_layers(self):
        activation_dict = {
            'relu':F.relu,
            'elu':F.elu,
            'gelu':F.gelu,
            'leaky_relu':F.leaky_relu,
        }
        self.activation = activation_dict['elu']
        self._setup_GNN_layer()
        self._setup_BN_layer()
        self._setup_Aggregatting_layer()
        if self.model_settings['rnn_type'] != 'Informer':
            self._setup_cascade_embedding_layer()
            self._setup_RNN_layer()
            self._setup_time_decay_layer()
            self._setup_output_layer()
        else:
            self._setup_informer_layers()
            # self._setup_vae_layer()
        

    def forward(self, data):
        
        '''data.x'''
        # (batch_size *  number_of_snapshot(uncertain) * node_num_in_snapshot(uncertain), node_input_feature_dim)
        '''data.edge_index'''
        # (2 , batch_size *  number_of_snapshot(uncertain) * edges_num_in_snapshot(uncertain) )
        batch_size = data.y.shape[0]
        # batch_size <- (batch_size)[0]
        batch_index = data.batch
        # batch_size *  number_of_snapshot(uncertain) * node_num_in_snapshot(uncertain)
        x_time_interval = data.x_time_interval
        # batch_size *  number_of_snapshot(uncertain)
        x_used_snapshot_num_in_cascade = data.x_used_snapshot_num_in_cascade
        # batch_size
        x_struct_time_features = data.x_struct_time_features
        # (batch_size *  number_of_snapshot(uncertain), time_feature_input_dim)
        x_snapshots_features = data.x_snapshots_features
        # (batch_size *  number_of_snapshot(uncertain), cascade_feature_input_dim)

        device = batch_index.device

        # GNN
        if self.train_settings['stable']:
            x = self.GNN_layer(data.x, 
                    SparseTensor(row=data.edge_index[0], 
                                    col=data.edge_index[1], 
                                    sparse_sizes=(data.num_nodes, data.num_nodes)).t())         # deteministic version
        else:
            x = self.GNN_layer(data.x, data.edge_index)
        
        # x = self.batch_norm_layer_1(x) if not self.train_settings['use_multi_gpu'] else x
        # (batch_size *  number_of_snapshot(uncertain) * node_num_in_snapshot(uncertain), GNN_out_dim)
        train_logger.debug('after GNN')
        train_logger.debug(x)

        # padding后输入到动态路由
        # 这几步可能会爆显存
        try:
            train_logger.debug('GPU memory before extend')
            train_logger.debug(torch.cuda.memory_allocated(device))
            train_logger.debug(torch.cuda.memory_reserved(device))
            train_logger.debug(torch.cuda.max_memory_allocated(device))
            # padding后再输入到动态路由
            tmp = []
            for i in range(batch_size):
                each_cascade_index = (batch_index == i).nonzero()
                # 按级联分组，级联内部按快照分割
                each_cascade = x.index_select(0, each_cascade_index.flatten()).view(x_used_snapshot_num_in_cascade[i], -1
                                                                        , self.model_settings['gnn_out_channels']).chunk(x_used_snapshot_num_in_cascade[i], 0)
                # 压缩维度后存入统一的一个列表，每个元素对应一个快照中的节点特征
                tmp.extend(list(snapshot.squeeze(0) for snapshot in each_cascade))
            
            train_logger.debug('GPU memory after extend')
            train_logger.debug(torch.cuda.memory_allocated(device))
            train_logger.debug(torch.cuda.memory_reserved(device))
            train_logger.debug(torch.cuda.max_memory_allocated(device))
            
            # padding补零
            tmp = pad_sequence(tmp, batch_first=True)
            # (sum(num_of_snapshot), max(node_number), GNN_out_dim)

            train_logger.debug('GPU memory after padding')
            train_logger.debug(torch.cuda.memory_allocated(device))
            train_logger.debug(torch.cuda.memory_reserved(device))
            train_logger.debug(torch.cuda.max_memory_allocated(device))
            
            # 输入到动态路由，聚合节点特征为级联快照特征
            x = self.aggregate_layer(tmp)
            # x = self.batch_norm_layer_2(x) if not self.train_settings['use_multi_gpu'] else x
            # (sum(num_of_snapshot), Dynamic_routing_out_dim)
            train_logger.debug('after dynamic routing')
            train_logger.debug(x)

            
            train_logger.debug('GPU memory after dynamic routing, before append')
            train_logger.debug(torch.cuda.memory_allocated(device))
            train_logger.debug(torch.cuda.memory_reserved(device))
            train_logger.debug(torch.cuda.max_memory_allocated(device))

        except RuntimeError as exception:
            if "CUDA out of memory" in str(exception):
                train_logger.debug(str(exception))
                del tmp
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                # batch中有极大级联导致爆显存
                # 不再padding后统一输入
                # 依次遍历后输入
                tmp = []
                for i in range(batch_size):
                    each_cascade_index = (batch_index == i).nonzero()
                    # 按级联分组，级联内部按快照分割
                    each_cascade = x.index_select(0, each_cascade_index.flatten()).view(x_used_snapshot_num_in_cascade[i], -1
                                                                            , self.model_settings['gnn_out_channels'])
                    # 直接输入到动态路由
                    each_cascade = self.aggregate_layer(each_cascade)
                    # number_of_snapshot_in_current_cascade * Dynamic_routing_out_dim

                    # 直接相加
                    # each_cascade = torch.sum(each_cascade, dim=1)
                    # number_of_snapshot_in_current_cascade * Dynamic_routing_out_dim

                    tmp.extend(each_cascade.chunk(x_used_snapshot_num_in_cascade[i], 0))
                
                x = torch.cat( tuple(tmp), dim=0)
                # x = torch.matmul(x, torch.randn(self.model_settings['gnn_out_channels'], self.model_settings['capsule_out_dim'], device=device))

            else:
                raise exception


        # 按照快照所属级联（每个级联即为batch中的一个数据）重新组装成多个级联的信息
        tmp = []
        batch_x_time_interval = []
        batch_x_struct_time_features = []
        batch_x_cascade_snapshot_features = []
        used_index = 0
        for i in x_used_snapshot_num_in_cascade:
            tmp.append( x[used_index:used_index+i] )
            batch_x_time_interval.append( x_time_interval[used_index:used_index+i] )
            batch_x_struct_time_features.append( x_struct_time_features[used_index:used_index+i] )
            batch_x_cascade_snapshot_features.append( x_snapshots_features[used_index:used_index+i] )
            used_index += i
        
        train_logger.debug('GPU memory after append')
        train_logger.debug(torch.cuda.memory_allocated(device))
        train_logger.debug(torch.cuda.memory_reserved(device))
        train_logger.debug(torch.cuda.max_memory_allocated(device))
        
        # padding前
        # tmp_batch_cascade: (batch_size, number_of_snapshot_in_each_cascade(uncertain), Dynamic_routing_out_dim)
        # batch_x_time_interval: (batch_size,  number_of_snapshot_in_each_cascade(uncertain))
        # batch_x_struct_time_features: (batch_size,  number_of_snapshot_in_each_cascade(uncertain), time_feature_input_dim)
        
        # 再次padding
        x = pad_sequence(tmp, batch_first=True)
        # (batch_size, max(num_of_snapshot), Dynamic_routing_out_dim)
        batch_x_time_interval = pad_sequence(batch_x_time_interval, batch_first=True)
        # (batch_size,  max(number_of_snapshot))
        batch_x_struct_time_features = pad_sequence(batch_x_struct_time_features, batch_first=True)
        # (batch_size,  max(number_of_snapshot), time_feature_input_dim)
        batch_x_cascade_snapshot_features = pad_sequence(batch_x_cascade_snapshot_features, batch_first=True)
        # (batch_size,  max(number_of_snapshot), cascade_feature_input_dim)

        train_logger.debug('GPU memory after padding')
        train_logger.debug(torch.cuda.memory_allocated(device))
        train_logger.debug(torch.cuda.memory_reserved(device))
        train_logger.debug(torch.cuda.max_memory_allocated(device))
        
        # 输入到RNN
        if self.model_settings['rnn_type'] != 'Informer':
            y = data.y_tmp
            
            if self.data_settings['use_cascade_level_features_in_rnn']:
                # 级联级别特征编码
                batch_x_cascade_snapshot_features = self.cascade_embedding_layer(batch_x_cascade_snapshot_features)
                # (batch_size, max(num_of_snapshot), cascade_level_features_dim)
                train_logger.debug('batch_x_cascade_snapshot_features after embedding')
                train_logger.debug(batch_x_cascade_snapshot_features)

                # concatenate，表示最终级联
                x = torch.cat( (x, batch_x_cascade_snapshot_features), dim=2)
                # (batch_size, max(num_of_snapshot), Dynamic_routing_out_dim + cascade_level_features_dim)
                train_logger.debug('batch_cascade after concatenate')
                train_logger.debug(x)

            # 输入到RNN
            if self.train_settings['use_multi_gpu']:
                self.RNN_layer.flatten_parameters()
            x = self.RNN_layer(x, x_used_snapshot_num_in_cascade)
            # (batch_size, max(num_of_snapshot), Dynamic_routing_out_dim)
            train_logger.debug('batch_cascade after rnn')
            train_logger.debug(x)

            # 时间衰减层
            x = self.TimeDecay_layer(x, batch_x_time_interval).sum(1)
            #  after sum                                 before sum
            # (batch_size, Dynamic_routing_out_dim)  <- (batch_size, num_of_snapshot, Dynamic_routing_out_dim)
            x = self.batch_norm_layer_3(x) if not self.train_settings['use_multi_gpu'] else x
            train_logger.debug('batch_cascade after timedecay')
            train_logger.debug(x)

            # MLP预测增量
            x = self.output_layer(x)
            x = F.elu(x, inplace=False)
            x = x.squeeze(-1)
            train_logger.debug('prediction after MLP')
            train_logger.debug(x)

            train_logger.debug('ground truth is ')
            train_logger.debug(y)

            return x, y
        
        # 或输入到Informer
        else:
            # 取出data
            y_sequence_length = data.y_predict_targets.view(batch_size, -1).size(1)
            #  decoder_input_length  <- (batch_size * decoder_input_length) [1]
            y_struct_time_features = data.y_struct_time_features.view(batch_size, y_sequence_length, -1)
            # (batch_size , decoder_input_length, time_feature_input_dim)   <-   (batch_size *  decoder_input_length, time_feature_input_dim)
            y_snapshots_features = data.y_snapshots_features.view(batch_size, y_sequence_length, -1)
            # (batch_size *  number_of_snapshot(uncertain), cascade_feature_input_dim)   <-   (batch_size *  decoder_input_length, cascade_feature_input_dim)
            y = data.y_tmp.view(batch_size, y_sequence_length)
            # y_predict_targets = data.y_predict_targets.view(batch_size, y_sequence_length)
            # (batch_size, decoder_input_length)
            y_used_snapshot_in_prediction = data.y_used_snapshot_in_prediction.view(batch_size, -1)
            # (batch_size, number_of_snapshot(certain))
            y_time_interval = data.y_time_interval.view(batch_size, y_sequence_length)
            # (batch_size, decoder_input_length)

            # 输入到encoder前先从中取出要输入到decoder的部分，因为之后输入到Encoder前会被编码
            tmp = []
            for i in range(batch_size):
                snapshot_pos = (y_used_snapshot_in_prediction[i] == 1).nonzero()
                cas_snap = x[i].index_select(0, snapshot_pos.flatten())
                tmp.append(cas_snap)
            decoder_x = pad_sequence(tmp, batch_first=True)
            # (batch_size, max(num_of_snapshot) / 2, Dynamic_routing_out_dim)
            
            # version1：补零后concat
            # 这里有问题，全为0后只有时间特征，不同级联的时间特征完全一致，导致输出也一致
            padding_zero = torch.zeros( x.size(0),
                                        y_sequence_length - decoder_x.size(1) , 
                                        x.size(2), device=device
                                        )
            decoder_x = torch.cat( (decoder_x, padding_zero), dim = 1)
            # (batch_size, max(num_of_snapshot), Dynamic_routing_out_dim)
            
            # version2：直接复制一份，交由时间衰减层处理
            # padded_decoder_input = padded_decoder_input.repeat(1,2,1)
            # (batch_size, max(num_of_snapshot), Dynamic_routing_out_dim)

            # version3：交给vae处理，我真是个天才
            # vae_padded_decoder_input = self.vae_layer(padded_decoder_input)[0]
            # padded_decoder_input = torch.cat((padded_decoder_input, vae_padded_decoder_input), dim=1)
            # (batch_size, max(num_of_snapshot), Dynamic_routing_out_dim)

            train_logger.debug('decoder_input before embedding')
            train_logger.debug(decoder_x)

            # batch_norm
            train_logger.debug('decoder_input after batch_norm')
            train_logger.debug(decoder_x)

            train_logger.debug('cascade before input to Informer')
            train_logger.debug(x)
            train_logger.debug('padded_x_batch_time_interval before input to Informer')
            train_logger.debug(batch_x_time_interval)
            train_logger.debug('padded_x_batch_struct_time_features before input Informer')
            train_logger.debug(batch_x_struct_time_features)
            
            # 输入到Encoder前再次进行embedding
            # 时间衰减层用于位置编码，timefeature层添加时间信息
            x = self.informer_embedding_layer(x, 
                                                        batch_x_time_interval,
                                                        batch_x_struct_time_features, 
                                                        batch_x_cascade_snapshot_features,
                                                        padding = True)
            x = self.informer_batch_norm_layer_1(x)
            # (batch_size, max(num_of_snapshot), Dynamic_routing_out_dim)
            train_logger.debug('cascade after embedding')
            train_logger.debug(x)

            # 输入到encoder
            # 我用错了！？我把要输入到decoder的部分输入到encoder里了，小样本上效果居然挺好？
            encoder_out, attn = self.informer_encoder_layer(x)
            train_logger.debug('cascade after encoder')
            train_logger.debug(encoder_out)
            train_logger.debug('attention from encoder')
            train_logger.debug(attn)

            train_logger.debug('y_time_interval before embedding')
            train_logger.debug(y_time_interval)
            train_logger.debug('y_struct_time_features before embedding')
            train_logger.debug(y_struct_time_features)

            # 待预测序列位置编码
            # padding_zero = self.TimeDecay_layer(padding_zero, y_time_interval)
            # 待预测序列时间编码
            # y_struct_time_features = self.informer_timeEmbedding_layer(y_struct_time_features)

            # 整合，一次embedding
            decoder_x = self.informer_embedding_layer(decoder_x, y_time_interval,
                                                        y_struct_time_features, 
                                                        y_snapshots_features,
                                                        padding = False)
            decoder_x = self.informer_batch_norm_layer_2(decoder_x)
            train_logger.debug('decoder_input after embedding')
            train_logger.debug(decoder_x)
            

            # 输入到decoder
            x = self.informer_decoder_layer(decoder_x, encoder_out)

            train_logger.debug('decoder_out after decoder')
            train_logger.debug(x)

            # 输出预测结果
            
            x = self.informer_output_layer(x).squeeze()

            train_logger.debug('prediction after linear')
            train_logger.debug(x)

            train_logger.debug('ground truth is ')
            train_logger.debug(y)

            # 这里有问题，训练时应当用所有预测值计算loss，而非仅使用预测部分
            return x.view(batch_size,-1), y
            # return prediction.reshape(-1,1).squeeze(0), y_predict_targets[:,-self.data_settings['predict_part_length']:].reshape(-1,1).squeeze(0)
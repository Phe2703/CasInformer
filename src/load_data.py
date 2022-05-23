import time, datetime, calendar
from typing import List,Tuple
from collections import defaultdict
import random
import json
import logging
import multiprocessing

import torch
from torch.utils.data import Dataset as DatasetPytorch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
import networkx as nx
from tqdm import tqdm

from config import dataset_path,tmp_path
from train_utils import trans_time_feature
from pub_utils import data_jsonl_loader, data_jsonl_saver, get_process_num


train_logger = logging.getLogger('models')

class CasInformerDataListPyG:
    '''
    将cascade data转为pyg的data
    不保存到本地因此不继承DataSet
    add: 保存了数据对应索引及统计信息
    update:处理数据还是浪费时间，改为使用pyg的Dataset
    '''
    def __init__(self, dataset_name: str, data_settings: dict, model_settings: dict, train_settings: dict, pre_load = False):
        '''
        @para
        pre_load_all: 预先加载并处理好每一个数据，转化为PyG的Data对象再返回
        '''
        self.dataset_name = dataset_name
        self.data_settings = data_settings
        self.model_settings = model_settings
        self.train_settings = train_settings
        self.pre_load_all = pre_load
        # super().__init__()
        self.load_data()
    
    def __getitem__(self, index) -> Data:
        if self.pre_load_all:
            cascade_data = self.data[index]
        else:
            cascade_data = extract_cascade_features(self.data[index], self.data_settings, self.model_settings)
            cascade_data = features_to_Data(cascade_data)
        return adjust_predict_target(cascade_data, self.train_settings['predict_type'], self.model_settings['rnn_type'])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def load_data(self):
        '''
        数据加载过程：
        读取本地数据数据集 -> 根据数据范围限制过滤数据并将过滤后的信息（数据索引）保存到本地
        -> 对过滤后的数据进行处理 -> 转为tensor -> 转为PyG的Data
        '''
        all_cascade_data = data_jsonl_loader(self.dataset_name, self.dataset_name)
        index_filename = tmp_path / self.dataset_name / f'{data_limit_str(self.data_settings)}.json'
        if index_filename.exists():
            print('loading previous index......')
            data_to_use = json.load(index_filename.open(encoding='utf8'))
            data_to_drop = [i for i in range(len(all_cascade_data)) if str(i) not in data_to_use.keys()]
            data_to_drop.sort(reverse=True)

            for k,v in data_to_use.items():
                cascade_data = all_cascade_data[int(k)]
                cascade_data['last_observe_snapshot_index'] = v[0]
                cascade_data['last_predict_snapshot_index'] = v[1]
            
            for i in data_to_drop:
                all_cascade_data.pop(i)
            
            self.data = all_cascade_data

        else:
            print('filted data index unfound, starting filt data ......')
            self.data, data_to_use = cascade_data_filter(all_cascade_data, self.data_settings)
            # index_filename.write_text(json.dumps(data_to_use), encoding='utf8')
        
        if self.pre_load_all:
            print('pre generating all data features')
            self.data = trans_all_cascade(self.data, self.data_settings, self.model_settings)
        
        if self.data_settings['sort_by_time']:
            self.data.sort(key = lambda x: int(x['cascade_id']))
            

class CasInformerDataSetPyG(InMemoryDataset):
    def __init__(self, dataset_name: str, data_settings: dict, model_settings: dict, train_settings: dict,
                    pre_load = True,root = tmp_path, transform = None, pre_transform = None, pre_filter = None):
        '''
        transform函数在访问之前动态地转换数据对象(因此最好用于数据扩充)
        pre_transform函数在将数据对象保存到磁盘之前应用转换(因此它最好用于只需执行一次的大量预计算)
        pre_filter函数可以在保存之前手动过滤掉数据对象。用例可能涉及数据对象属于特定类的限制
        '''
        self.dataset_name = dataset_name
        self.data_settings = data_settings
        self.model_settings = model_settings
        self.train_settings = train_settings

        def transform_wrapper(data):
            return adjust_predict_target(data, self.train_settings['predict_type'], self.model_settings['rnn_type'])
        def pre_transform_wrapper(data):
            return features_to_Data(extract_cascade_features(data, self.data_settings, self.model_settings))
        def pre_filter_wrapper(data):
            return cascade_is_valid(data, self.data_settings)

        super().__init__(str(root / dataset_name), transform_wrapper, pre_transform_wrapper, pre_filter_wrapper)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return f'{self.dataset_name}.jsonl'

    @property
    def processed_file_names(self):
        return f'{data_limit_str(self.data_settings)}_{feature_limit_str(self.data_settings)}.pt'
    
    def process(self):
        # 加载原始数据集
        data_list = data_jsonl_loader(self.dataset_name, self.dataset_name)
        print('processed data unfound, ', end=', ')
        
        # 过滤
        print('filtering......', end=', ')
        if self.pre_filter is not None:
            data_list = [data for data in tqdm(data_list) if self.pre_filter(data)]

        # 转换
        print('processing......', end=', ')
        if self.pre_transform is not None:
            # data_list = [self.pre_transform(data) for data in data_list]
            data_list = trans_all_cascade(data_list, self.data_settings, self.model_settings)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    

def data_limit_str(data_settings) -> str:
    '''
    根据数据范围限制生成对应的字符串标记
    '''
    processed_filename = [data_settings['observe_time'],
                        data_settings['predict_time'],
                        data_settings['min_size_in_observe_window'],
                        data_settings['max_size_in_observe_window'],
                        data_settings['publish_start'],
                        data_settings['publish_end'],
                        ]
    
    return '_'.join([str(int(i)) for i in processed_filename])

def feature_limit_str(data_settings) -> str:
    '''
    根据所使用特征范围限制生成对应的字符串标记
    '''
    processed_filename = [
                        data_settings['dim_of_state_features'] if data_settings['use_node_state_feature'] else 0,
                        data_settings['dim_of_degree_features'] if data_settings['use_node_degree_feature'] else 0,
                        data_settings['dim_of_time_features'] if data_settings['use_node_time_feature'] else 0,
                        data_settings['dim_of_depth_features'] if data_settings['use_node_depth_feature'] else 0,

                        data_settings['dim_of_cascade_depth'] if data_settings['use_cascade_depth'] else 0,
                        data_settings['dim_of_cascade_activated_size'] if data_settings['use_cascade_activated_size'] else 0,
                        data_settings['dim_of_cascade_time'] if data_settings['use_cascade_time'] else 0,

                        data_settings['time_tuple_features_dim'],
                        ]
    end = ''.join([str(int(i)) for i in processed_filename])
    number_of_snapshot = data_settings['number_of_snapshot']
    
    return f'{number_of_snapshot}_{end}'

def adjust_predict_target(cascade: Data, predict_type: str, rnn_type: str) -> Data:
    # return cascade
    if rnn_type == 'Informer':
        if predict_type == 'full':
            cascade.y_tmp = cascade.y_predict_targets
        else:
            cascade.y_tmp = cascade.y_predict_targets - cascade.observed_cascade_size
    else:
        cascade.y_predict_targets = None
        cascade.y_struct_time_features = None
        cascade.y_snapshots_features = None
        cascade.y_used_snapshot_in_prediction = None
        cascade.y_time_interval = None
        if predict_type == 'full':
            cascade.y_tmp = cascade.y
        else:
            cascade.y_tmp = cascade.y - cascade.observed_cascade_size
    
    return cascade

def features_to_tensor(all_snapshot_node_features: List[List], 
                            all_snapshot_edges: np.array, 
                            predict_target: int, 
                            x_snapshot_time_interval: List[float], 
                            x_used_snapshot_num_in_cascade: List[int],
                            x_struct_time_features: List[List[int]],
                            y_struct_time_features: List[List[int]],
                            y_predict_targets: List[int],
                            y_used_snapshot_in_prediction: List[int],
                            y_time_interval: List[float],
                            x_snapshots_features: List[List[int]],
                            y_snapshots_features: List[List[int]],
                            y_predict_series_pos:int,
                            observed_cascade_size:int,
                            cascade_id:int,
                        ) -> Tuple[torch.Tensor]:
    '''
    数据类型转换, 由python内置数据类型转为tensor
    '''
    return ( 
                torch.Tensor(np.array(all_snapshot_node_features)).to(torch.float32), 
                # torch.tensor(np.array(all_edges_to_snapshot_feature).T).to(torch.int64), 
                torch.t(torch.tensor(np.array(all_snapshot_edges)).to(torch.int64)), 
                torch.tensor(predict_target).float(), 
                torch.tensor(x_snapshot_time_interval), 
                torch.tensor(x_used_snapshot_num_in_cascade),
                torch.tensor(x_struct_time_features),
                torch.tensor(y_struct_time_features),
                torch.tensor(y_predict_targets).float(),
                torch.tensor(y_used_snapshot_in_prediction),
                torch.tensor(y_time_interval),
                torch.tensor(x_snapshots_features).float(),
                torch.tensor(y_snapshots_features).float(),
                torch.tensor(y_predict_series_pos).float(),
                torch.tensor(observed_cascade_size).float(),
                torch.tensor(cascade_id).float(),
            )

def features_to_Data(cascade_features) -> Data:
    '''
    数据类型转换, tensor -> PyG.Data
    '''
    cascade_features = features_to_tensor(*cascade_features)
    node_features = cascade_features[0]
    edges = cascade_features[1]
    predict_target = cascade_features[2]
    x_snapshot_time_interval = cascade_features[3]
    x_used_snapshot_num_in_cascade = cascade_features[4]
    x_struct_time_features = cascade_features[5]
    y_struct_time_features = cascade_features[6]
    y_predict_targets = cascade_features[7]
    y_used_snapshot_in_prediction = cascade_features[8]
    y_time_interval = cascade_features[9]
    x_snapshots_features = cascade_features[10]
    y_snapshots_features = cascade_features[11]
    y_predict_series_pos = cascade_features[12]
    observed_cascade_size = cascade_features[13]
    cascade_id = cascade_features[14]
    
    return Data(
                x = node_features, 
                edge_index = edges, 
                y = predict_target, 
                x_time_interval = x_snapshot_time_interval, 
                x_used_snapshot_num_in_cascade = x_used_snapshot_num_in_cascade,
                x_struct_time_features = x_struct_time_features,
                y_struct_time_features = y_struct_time_features,
                y_predict_targets = y_predict_targets,
                y_used_snapshot_in_prediction = y_used_snapshot_in_prediction,
                y_time_interval = y_time_interval,
                x_snapshots_features = x_snapshots_features,
                y_snapshots_features = y_snapshots_features,
                y_predict_series_pos = y_predict_series_pos,
                observed_cascade_size = observed_cascade_size,
                cascade_id = cascade_id,
            )

def cascade_data_filter(all_cascade_data: List[dict], data_settings:dict) -> List[dict]:
    '''
    按配置过滤数据，并保存过滤时记录的信息
    '''
    data_to_drop = []
    data_to_use = {}
    # 记录有效级联数据的观察时间对应快照索引，以及观察时间到预测时间的级联增量
    # 同时记录哪些数据无效需要丢弃
    for i, cascade_data in enumerate(tqdm(all_cascade_data)):
        if cascade_is_valid(cascade_data, data_settings):
            data_to_use[i] = [cascade_data['last_observe_snapshot_index'], 
                            cascade_data['last_predict_snapshot_index'],
                            ]
        else:
            data_to_drop.insert(0, i)

    for i in data_to_drop:
        all_cascade_data.pop(i)
    
    # if data_settings['sort_by_time']:
    #     all_cascade_data.sort(key = lambda x: x['time_for_sort'])
    print(f'number of valid cascade : {len(all_cascade_data)} / {len(all_cascade_data) + len(data_to_drop)}')
    return all_cascade_data, data_to_use

def cascade_is_valid(cascade_data, data_settings:dict) -> bool:
    '''
    判断级联是否符合过滤条件
    '''
    # 检查级联时间范围
    if data_settings["publish_start"] and data_settings["publish_start"] > int(cascade_data['time_for_filter']):
        # data_to_drop.insert(0, i)
        train_logger.debug(f'drop cascade {cascade_data["cascade_id"]}, time {cascade_data["time_for_filter"]} < start time {data_settings["publish_start"]}')
        return False
    if data_settings['publish_end'] and data_settings["publish_end"] < int(cascade_data['time_for_filter']):
        # data_to_drop.insert(0, i)
        train_logger.debug(f'drop cascade {cascade_data["cascade_id"]}, time {cascade_data["time_for_filter"]} > end time {data_settings["publish_end"]}')
        return False
    # 检查快照数量是否符合
    # (只有起始时刻一个快照的数据也会被过滤，不是bug，因为这样没有预测的意义)
    last_observe_snapshot_index =  get_last_observe_snapshot_index(cascade_data, data_settings)
    if last_observe_snapshot_index:
        # cascade_data['snapshot_indices'] = set(snapshot_indices)
        cascade_data['last_observe_snapshot_index'] = last_observe_snapshot_index
    else:
        # data_to_drop.insert(0, i)
        train_logger.debug(f'drop cascade {cascade_data["cascade_id"]}, last_observe_snapshot_index is {last_observe_snapshot_index}')
        return False
    # 检查级联大小范围
    if data_settings["predict_time"]:
        # 部分数据的流行度计算不包含第一次发布，因此发布路径记录数会比统计级联大小多1，如aps
        # 部分数据包含root节点，如weibo
        # 自己的数据处理中包含第一次发布
        size_bias = cascade_data['size_bias']
        f_index = [i for i,t in enumerate(cascade_data['snapshot_times']) if t <= data_settings["predict_time"]][-1]
        cascade_data['last_predict_snapshot_index'] = f_index
        as_final_size = cascade_data[f'snapshot_{f_index}']['activated_size'] - size_bias
        observed_cascade_size = cascade_data[f'snapshot_{last_observe_snapshot_index}']['activated_size'] - size_bias
        
        if data_settings['min_size_in_observe_window'] and data_settings['min_size_in_observe_window'] > observed_cascade_size:
            # data_to_drop.insert(0, i)
            train_logger.debug(f'drop cascade {cascade_data["cascade_id"]}, size {observed_cascade_size} < {data_settings["min_size_in_observe_window"]}')
            return False
        if data_settings['max_size_in_observe_window'] and data_settings['max_size_in_observe_window'] < observed_cascade_size:
            # data_to_drop.insert(0, i)
            train_logger.debug(f'drop cascade {cascade_data["cascade_id"]}, size {observed_cascade_size} > {data_settings["max_size_in_observe_window"]}')
            return False
        
        # 预测最终时刻的规模
        # predict_target = as_final_size
        # 或预测观察时刻到最终时刻的增量
        # predict_target = as_final_size - observed_cascade_size       # 预测增量可能会出现增量为0的情况，容易出现nan

        # cascade_data['predict_target'] = predict_target          
    
    # train_logger.debug(f"time:{cascade_data['time_for_filter']}, size:{cascade_data[f'snapshot_{last_observe_snapshot_index}']['activated_size']}")
    
    return True

def trans_all_cascade(all_data, data_settings, model_settings, multiprocess = True, threshold = 10000):
    if multiprocess and len(all_data) > threshold:
        # 多进程并行
        return [ features_to_Data(item) for item in multi_process_extract_cascade_features(all_data, 
                                                                data_settings, model_settings)]
    else:
        # 单进程遍历
        return  [ features_to_Data(extract_cascade_features(item,
                                data_settings, model_settings)) for item in tqdm(all_data)]
       
def multi_process_extract_cascade_features(all_cascade_data: List[dict], 
                                            data_settings: dict, 
                                            model_settings: dict) -> List[Tuple]:
    '''
    多进程调用特征提取函数
    '''
    process_num = get_process_num()
    with multiprocessing.Pool(processes=process_num, 

                            ) as pool:
        all_cascade_features = list(pool.imap(extract_cascade_features, 
                                        ( (cas, data_settings, model_settings) for cas in tqdm(all_cascade_data)), 
                                        chunksize=1),
                                )
    return all_cascade_features

def extract_cascade_features(cascade_data: dict, data_settings: dict = None, 
                    model_settings: dict = None) -> Tuple[List[List],List[List],int,List[int],int]:
    '''
    从一个cascade data提取要输入模型的特征
    '''
    # 多进程输入的是元组，先解包
    if isinstance(cascade_data, tuple) and len(cascade_data) == 3:
        cascade_data, data_settings, model_settings = cascade_data
    
    cascade_id = int(cascade_data['cascade_id'])
    size_bias = cascade_data['size_bias']
    last_predict_snapshot_index = cascade_data["last_predict_snapshot_index"]
    last_observe_snapshot_index = cascade_data['last_observe_snapshot_index']   # 观察时刻对应的快照索引
    as_final_size = cascade_data[f'snapshot_{last_predict_snapshot_index}']['activated_size'] - size_bias
    observed_cascade_size = cascade_data[f'snapshot_{last_observe_snapshot_index}']['activated_size'] - size_bias
    # 预测最终时刻的规模
    predict_target = as_final_size
    # 或预测观察时刻到最终时刻的增量
    # predict_target = as_final_size - observed_cascade_size       # 预测增量可能会出现增量为0的情况，容易出现nan

    # if predict_target == 0 or predict_target == 1:
    #     print(cascade_data)

    # 先遍历一遍观察时刻之前的快照，统计观察时刻前出现的所有节点
    # 为了节省内存没有直接记录到文件里，除非内存很大，否则不要修改
    all_snapshot_nodes_before_observe = set()
    for i in range(len(cascade_data['snapshot_times'])):
        if i > last_observe_snapshot_index:
            break
        key = f'snapshot_{i}'
        snapshot_new_edges = cascade_data[key]['new_edges']
        for s,t in snapshot_new_edges:
            all_snapshot_nodes_before_observe.add(s)
            all_snapshot_nodes_before_observe.add(t)
    num_of_nodes = len(all_snapshot_nodes_before_observe)
    node_map = { str(node_id):i for i, node_id in enumerate(all_snapshot_nodes_before_observe) }

    # 遍历准备，记录节点特征
    all_snapshot_node_features = []
    cascade_edges = set()
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    node_present_time_dict = {}
    source_node = cascade_data['source_node']
    if data_settings['use_node_depth_feature']:
        cascade_graph = nx.DiGraph()
        cascade_graph.add_node(source_node)

    x_struct_time_features = []
    y_struct_time_features = []
    y_predict_targets = []
    y_time_interval = []
    x_snapshots_features = []
    y_snapshots_features = []

    # 仅按照指定数量取快照特征
    snapshot_indices = get_observe_snapshot_index(cascade_data, data_settings)
    # 输入到decoder中的待预测序列包含观察时刻的末尾部分
    expect_length = data_settings['expect_length']
    label_part, predict_part, y_used_snapshot_in_prediction = gen_predict_snapshot_index(cascade_data, data_settings, snapshot_indices)
    x_snapshot_time_interval = []
    # 遍历观察时刻之前的快照
    for i in range(len(cascade_data['snapshot_times'])):
        if i > last_observe_snapshot_index:
            break
        key = f'snapshot_{i}'
        current_cascade_snapshot = cascade_data[key]
        # 处理快照数据
        
        # 入度和出度也应该根据快照时刻不同而不同
        # 更新入度与出度，同时记录快照包含的所有节点
        snapshot_new_edges = current_cascade_snapshot['new_edges']

        if data_settings['use_node_depth_feature']:
            cascade_graph.add_edges_from(snapshot_new_edges)
        
        for s,t in snapshot_new_edges:
            out_degree[s] += 1
            in_degree[t] += 1
            if (s not in node_present_time_dict.keys()):
                node_present_time_dict[s] = current_cascade_snapshot["occur_time"]
            if (t not in node_present_time_dict.keys()):
                node_present_time_dict[t] = current_cascade_snapshot["occur_time"]
        # 更新快照的边
        cascade_edges.update([tuple(edge) for edge in snapshot_new_edges])

        # 如果要取此时刻的快照
        if i in snapshot_indices:
            # print(i)
            # 级联级别特征
            # 快照时间
            x_struct_time_features.append(gen_snapshot_time_feature_by_timestamp(cascade_data['publish_time'],
                                        current_cascade_snapshot["occur_time"]))
            # 快照时间间隔
            x_snapshot_time_interval.append( float(current_cascade_snapshot["occur_time"] / data_settings['time_coefficient']) )
            # 快照特征，规模及深度
            cascade_snapshots_features = []
            if data_settings['use_cascade_activated_size']:
                cascade_snapshots_features.append( current_cascade_snapshot['activated_size'] - size_bias)
            if data_settings['use_cascade_depth']:
                cascade_snapshots_features.append( current_cascade_snapshot["max_depth"])
            if data_settings['use_cascade_time']:
                cascade_snapshots_features.append( current_cascade_snapshot["occur_time"] / data_settings['time_coefficient'] )
            x_snapshots_features.append(cascade_snapshots_features)
            
            # 节点级别特征
            # 级联快照的节点特征如下
            #         unactivate  activate  indegree outdegree  (occur time, current time, node_depth, max_depth, node_embedding)
            # node1 
            # node2 
            # node3
            
            # update： 将标记状态的0或1改为-1和1，避免填充0时混淆，节点出现时间同理，将第一个节点出现的时间从0改为-1
            # 初始化，默认节点均为未激活
            snapshot_node_features = defaultdict(list)
            for n in all_snapshot_nodes_before_observe:
                # 节点激活状态
                if data_settings['use_node_state_feature']:
                    snapshot_node_features[n].append(1)
                    snapshot_node_features[n].append(0)
                # 节点度特征
                if data_settings['use_node_degree_feature']:
                    snapshot_node_features[n].append(in_degree[n])
                    snapshot_node_features[n].append(out_degree[n])
                # 时间特征
                if data_settings['use_node_time_feature']:
                    snapshot_node_features[n].append(0)
                    snapshot_node_features[n].append(0)
                # 深度特征，节点距离源节点的最短路径以及二叉树形式的网络的层数，同样效果不佳
                if data_settings['use_node_depth_feature']:
                    snapshot_node_features[n].append(0)
            
            # 更新此时已出现节点的特征
            for n in node_present_time_dict.keys():
                pos = 0
                # 节点激活状态
                if data_settings['use_node_state_feature']:
                    snapshot_node_features[n][pos] = 0
                    snapshot_node_features[n][pos+1] = 1
                    pos += data_settings['dim_of_state_features']
                # 节点度特征
                if data_settings['use_node_degree_feature']:
                    # 度的字典是一个defaultdict
                    # 未出现的节点自动赋0，已出现节点赋对应值，因此无需再更新
                    pos += data_settings['dim_of_degree_features']
                # 节点时间特征
                if data_settings['use_node_time_feature']:
                    # snapshot_features[n][4] = node_present_time_dict[n] / data_settings['time_coefficient']
                    snapshot_node_features[n][pos] = node_present_time_dict[n] / data_settings['time_coefficient'] if node_present_time_dict[n] else -1  # 目前验证最佳
                    # snapshot_features[n][4] = ((data_settings['observe_time'] - node_present_time_dict[n]) / data_settings['time_coefficient']) + 1
                    # snapshot_features[n][5] = current_cascade_snapshot["occur_time"]
                    snapshot_node_features[n][pos+1] = current_cascade_snapshot["occur_time"] / data_settings['time_coefficient']                        # 目前验证最佳
                    # snapshot_features[n][5] = (data_settings['predict_time'] - current_cascade_snapshot["occur_time"]) / data_settings['time_coefficient']
                    pos += data_settings['dim_of_time_features']
                # 节点深度特征
                if data_settings['use_node_depth_feature']:
                    snapshot_node_features[n][pos] = nx.shortest_path_length(cascade_graph, source_node, n) + 1
                    # snapshot_features[n][7] = current_cascade_max_depth + 1
                    pos += data_settings['dim_of_depth_features']

            # 汇总节点特征
            all_snapshot_node_features.extend( snapshot_node_features.values() )

    # 遍历待预测序列中可观察时刻的快照
    # 为y生成对应特征
    for i in label_part:
        key = f'snapshot_{i}'
        current_cascade_snapshot = cascade_data[key]
        # 时间特征
        snapshot_time_features = gen_snapshot_time_feature_by_timestamp(cascade_data['publish_time'],
                                        current_cascade_snapshot["occur_time"])
        y_struct_time_features.append(snapshot_time_features)

        # 时间间隔
        y_time_interval.append( float(current_cascade_snapshot["occur_time"] / data_settings['time_coefficient']) )
        
        # 级联级别特征
        cascade_snapshots_features = []
        if data_settings['use_cascade_activated_size']:
            cascade_snapshots_features.append( current_cascade_snapshot['activated_size'] - size_bias)
        if data_settings['use_cascade_depth']:
            cascade_snapshots_features.append( current_cascade_snapshot["max_depth"])
        if data_settings['use_cascade_time']:
            cascade_snapshots_features.append( current_cascade_snapshot["occur_time"] / data_settings['time_coefficient'] )
        y_snapshots_features.append(cascade_snapshots_features)
        
        # ground truth
        y_predict_targets.append(current_cascade_snapshot['activated_size'] - size_bias)   # 预测规模
        # y_predict_targets.append(current_cascade_snapshot['activated_size'] - observed_cascade_size) # 或预测增量

    # 遍历待预测序列中不可观察时刻的快照
    # 只记录时间信息
    for i in predict_part:
        key = f'snapshot_{i}'
        current_cascade_snapshot = cascade_data[key]
        # 时间特征
        snapshot_time_features = gen_snapshot_time_feature_by_timestamp(cascade_data['publish_time'],
                                        current_cascade_snapshot["occur_time"])
        y_struct_time_features.append(snapshot_time_features)

        # 时间间隔
        y_time_interval.append( float(current_cascade_snapshot["occur_time"] / data_settings['time_coefficient']) )
        
        # 级联级别特征
        # 因为是预测时间，特征不可见，全部填零
        cascade_snapshots_features = []
        if data_settings['use_cascade_activated_size']:
            cascade_snapshots_features.append(0)
        if data_settings['use_cascade_depth']:
            cascade_snapshots_features.append(0)
        if data_settings['use_cascade_time']:
            cascade_snapshots_features.append(0)
            # cascade_snapshots_features.append(current_cascade_snapshot["occur_time"] / data_settings['time_coefficient'])
        y_snapshots_features.append(cascade_snapshots_features)
        
        # ground truth
        y_predict_targets.append(current_cascade_snapshot['activated_size'] - size_bias)   # 预测规模
        # y_predict_targets.append(current_cascade_snapshot['activated_size'] - observed_cascade_size) # 或预测增量
    
    # 遍历结束后，对于待预测序列，其长度可能仍达不到指定数量，因此要对其进行补全
    predict_indices = label_part + predict_part
    assert len(predict_indices) == len(y_predict_targets) == len(y_struct_time_features) == len(y_time_interval) == len(y_snapshots_features)
    mis_length = expect_length - len(y_predict_targets)
    
    if mis_length > 0:
        # 设置一个时间间隔
        last_one = float(cascade_data[f'snapshot_{predict_indices[-1]}']["occur_time"])
        if len(predict_indices) >= 2:
            second_last_one =  float(cascade_data[f'snapshot_{predict_indices[-2]}']["occur_time"])
            interval = last_one - second_last_one
        else:
            interval = last_one
        # 按照时间间隔填充缺失的位置
        for i in range(mis_length):
            
            new_one_time = last_one + interval * (i+1)
            y_struct_time_features.append( gen_snapshot_time_feature_by_timestamp(cascade_data['publish_time'], new_one_time) )
            y_time_interval.append( float(new_one_time / data_settings['time_coefficient']) )

            y_snapshots_features.append([0 for i in range(len(y_snapshots_features[-1]))])  # 填充0
            # y_snapshots_features.append( [0, 0, float(new_one_time / data_settings['time_coefficient']) ])
            # y_snapshots_features.append(y_snapshots_features[-1])                   # 还是填充末尾时刻的特征？
            
            y_predict_targets.append(y_predict_targets[-1])   # 规模视为不变

    # 检查
    assert len(y_predict_targets) == expect_length and len(y_struct_time_features) == expect_length and len(y_time_interval) == expect_length, 'predict length mismatch'

    # 编码时间特征
    x_struct_time_features = list(map(lambda stf: trans_time_feature(stf, end=data_settings['time_tuple_features_dim']), x_struct_time_features))
    y_struct_time_features = list(map(lambda stf: trans_time_feature(stf, end=data_settings['time_tuple_features_dim']), y_struct_time_features))

    # 遍历结束后快照的边即为完整级联的边
    # 由此生成边的邻接矩阵
    # 用作训练数据的snapshotfeatures有多少
    # 就要有对应数量的邻接矩阵（但只用一个邻接矩阵好像也没有影响）
    
    # all_snapshot_edges = [ [node_map[item[0]], node_map[item[1]]] for item in cascade_edges]
    
    all_snapshot_edges = []
    x_used_snapshot_num_in_cascade = int(len(all_snapshot_node_features) / num_of_nodes)
    # snapshot_edges = list(snapshot_edges).sort(key=lambda x:(x[0],x[1]))        # 排序仅仅是为了方便debug
    # 这一步的目的是对每个快照生成一个编号不同的边集合，使训练时gnn不会将多个快照混淆
    # 但其实好像不加也没什么影响
    for i in range(x_used_snapshot_num_in_cascade):
        all_snapshot_edges.extend( [ [(node_map[item[0]] + i*num_of_nodes),(node_map[item[1]] + i*num_of_nodes)] for item in cascade_edges] )
    
    return (all_snapshot_node_features, all_snapshot_edges, predict_target, 
            x_snapshot_time_interval, x_used_snapshot_num_in_cascade,
            x_struct_time_features, y_struct_time_features, 
            y_predict_targets, y_used_snapshot_in_prediction, y_time_interval,
            x_snapshots_features, y_snapshots_features, -mis_length-1,
            observed_cascade_size, cascade_id
            )

def get_last_observe_snapshot_index(cascade_data: dict, data_settings: dict) -> int:
    '''
    按照指定的观察时间选择对应索引
    '''
    snapshot_times = cascade_data['snapshot_times']
    # 如果指定了观察的时间
    if data_settings['observe_time']:
        threshold = data_settings['observe_time']
        # threshold = snapshot_times[-1] - data_settings['observe_time']
        
    # 否则根据观察时间比率设置
    # 勿用，参数中取消了这条
    # elif data_settings['observe_time_ratio']:
    #     threshold = snapshot_times[-1] * data_settings['observe_time_ratio']
    
    # 若无设置
    else:
        threshold = float('inf')
    
    candidate_indices = [i for i,t in enumerate(snapshot_times) if t <= threshold]
    return candidate_indices[-1] if candidate_indices else None
    # return candidate_indices[-number_of_snapshot:], 

def get_observe_snapshot_index(cascade_data: dict, data_settings: dict) -> List[int]:
    '''
    根据数据情况和指定的快照数量，生成对应索引
    '''
    number_of_snapshot = data_settings['number_of_snapshot']
    last_observe_snapshot_pos = cascade_data['last_observe_snapshot_index'] + 1
    candidate_indices = list(range(last_observe_snapshot_pos))
    if last_observe_snapshot_pos <= number_of_snapshot:
        return candidate_indices

    # 只取前n个，不合理，弃用
    # return candidate_indices[:number_of_snapshot] if number_of_snapshot else candidate_indices
    # 随机采样n个，不合适，弃用
    # return random.sample(range(last_observe_snapshot_index), 
    #     number_of_snapshot if last_observe_snapshot_index > number_of_snapshot else last_observe_snapshot_index)

    # v1: 取前k个，后面n个，中间等间隔m个
    first_k = 5
    last_n = 3
    mid_m = number_of_snapshot - first_k - last_n

    # v2：考虑到时间快照可能很密集，按时间段进行不同比例的选取
    # 前a%的时间，取k个
    # 后b%的时间，取n个
    # ps：现在觉得将v1中的last_n设为1即可
    # pss：现在觉得first_k也可以设为1，中间等间隔取10个
    # 因为间隔不是时间而是采样序列的编号，因此变化密集的时刻自然而然就采样更多了
    # update：实践效果不太好
    # first_k = 1
    # last_n = 1
    # mid_m = number_of_snapshot - first_k - last_n

    # 生成索引
    snapshot_indices = candidate_indices[:first_k]
    interval = (last_observe_snapshot_pos - first_k - last_n) // mid_m
    for i in range(1, mid_m + 1):
        snapshot_indices.append( candidate_indices[first_k + interval*i - 1])
    snapshot_indices.extend( candidate_indices[-last_n:])


    '''对比版本，等时间间隔选取快照'''
    # 等时间间隔选取快照
    # interval = data_settings['observe_time'] / number_of_snapshot
    # snapshot_times = cascade_data['snapshot_times']
    # snapshot_indices = []
    # for i in range(number_of_snapshot):
    #     tmp = [n for n,t in enumerate(snapshot_times) if t <= interval*(i+1)]
    #     if tmp:
    #         snapshot_indices.append(tmp[-1])
    '''仅用于消融实验时对比效果'''

    # if len(snapshot_indices) == 0:
    #     print(cascade_data)
    #     raise ValueError
    return snapshot_indices

def gen_predict_snapshot_index(cascade_data: dict, data_settings: dict, snapshot_indices: List[int]) -> List[int]:
    '''
    生成待预测时刻对应的索引
    '''
    label_part_length = data_settings['label_part_length']
    predict_part_length = data_settings['predict_part_length']
    snapshot_indices = list(set(snapshot_indices))
    snapshot_indices.sort()
    snapshot_num = len(snapshot_indices)
    
    last_observe_snapshot_pos = cascade_data['last_observe_snapshot_index']
    last_predict_snapshot_pos = cascade_data['last_predict_snapshot_index']
    
    crap = last_predict_snapshot_pos - last_observe_snapshot_pos
    if crap <= predict_part_length:
        label_part = snapshot_indices[-label_part_length:]
        predict_part = list(range(last_observe_snapshot_pos + 1, last_predict_snapshot_pos + 1))
    else:
        # 跟快照采样点保持一致
        label_intervel = snapshot_num // label_part_length
        if label_intervel == 0:
            label_part = snapshot_indices[-label_part_length:]
        else:
            label_part = list([snapshot_indices[-(1+i*label_intervel)] for i in range(label_part_length)][::-1])
            # label_part = list([last_observe_snapshot_pos - label_intervel * i for i in range(label_part_length)])[::-1]
        predict_interval = crap // predict_part_length
        # 确保末尾是预测时刻的索引，因此逆序遍历生成再翻转
        predict_part = list([last_predict_snapshot_pos - predict_interval * i for i in range(predict_part_length)])[::-1]
    
    used_snapshot_pos = [snapshot_indices.index(i) for i in label_part]
    used_snapshot_in_prediction = list([ 0 for i in range(data_settings['number_of_snapshot'])])
    for i in used_snapshot_pos:
        used_snapshot_in_prediction[i] = 1

    # predict_indices = label_part + predict_part
    return label_part, predict_part, used_snapshot_in_prediction
    # 这里有问题，遍历的时候待预测序列部分特征不可见，因此需要分开传参
    # return predict_indices, used_snapshot_in_prediction

def transdata(dataset_name: str, source_path = None, target_path = None):
    '''
    从deepHawkes提供的数据格式转化为自己需要的格式
    '''
    # <message_id>\tab<user_id>\tab<publish_time>\tab<retweet_number>\tab<retweets>
    # <message_id>:     the unique id of each message, ranging from 1 to 119,313.
    # <root_user_id>:   the unique id of root user. The user id ranges from 1 to 6,738,040.
    # <publish_time>:   the publish time of this message, recorded as unix timestamp.
    # <retweet_number>: the total number of retweets of this message within 24 hours.
    # <retweets>:       the retweets of this message, each retweet is split by " ". Within each retweet, it records 
    # the entile path for this retweet, the format of which is <user1>/<user2>/......<user n>:<retweet_time>.
    # retweets
    # 321825:0 
    # 321825/415484:2 
    # 321825/1514903:2 
    # 321825/1515794:2 
    # 321825/2089541:2 
    # 321825/3147556:2 
    # 321825/415484/2090434:3 
    # 321825/3147556/2090606:3 
    # 321825/3149360:3 
    # 321825/414762:4 
    # 321825/415238:4 
    # 321825/3147556/2090606/415281:4 
    # 321825/1515794/1514965:5 
    # 321825/3150433:6 
    # 321825/3147556/2090606/415281/2723288:7 
    # 321825/3147556/2090606/415281/415540:8 
    # 321825/3147556/2090606/415281/1515384:11 
    # 321825/3147556/2090606/415281/2723288/414962:16 
    # 321825/3147556/2090606/415281/2723288/414962/2723658:18  

    all_cascade_data = []
    # save_path = tmp_path / dataset_name
    # save_path.mkdir(exist_ok=True)
    # f_t = open(save_path / f"{dataset_name}.jsonl", 'w', encoding='utf8')
    if not source_path:
        source_path = dataset_path / dataset_name /'dataset.txt'

    if dataset_name =='twitter':
        key_time_type = 'd'      # 推特按日期筛选
    else:
        key_time_type = 'h'      # 微博按小时筛选

    with open(source_path, 'r') as f_s:
        # 每行对应一条级联
        for line in tqdm(f_s.readlines()):
            # 每个级联包含编号，源头用户id，发布时间，转发数量，及具体的转发信息
            message_id, user_id, publish_time, retweet_number, retweets = line.strip().split(maxsplit=4)

            # 将转发信息看作多个时刻的子级联
            sub_cascades = retweets.split()
            # 节点集合
            nodes_in_cascade = set()
            edges_in_cascade = set()
            # 记录每个时刻的信息，键为时间，值为对应时间的信息
            all_sub_state_dict = dict()
            total_size = 0
            for sc in sub_cascades:
                total_size += 1    # 每次转发都对应一条路径信息，级联规模计数+1
                sc_edges = set()
                # 参与节点
                # todo:同一个节点可能会重复转发，要考虑这种情况，目前未处理
                sc_nodes, snapshot_time = sc.split(':')
                snapshot_time = int(snapshot_time)
                sc_nodes = sc_nodes.split('/')
                # 遍历转发关系
                for i,n in enumerate(sc_nodes):
                    if i !=0 :
                        # 添加此时刻的有向边
                        sc_edges.add( (sc_nodes[i-1], n) )
                        edges_in_cascade.add( (sc_nodes[i-1], n) )
                    # 更新完整级联的节点集合
                    nodes_in_cascade.add(n)
                    # # 将此时刻的节点状态记为激活
                    # sub_state_dict[node] = 1
                
                # 更新级联快照字典
                if snapshot_time not in all_sub_state_dict.keys():
                    all_sub_state_dict[snapshot_time] = {
                                                "nodes":set(sc_nodes), 
                                                "edges":sc_edges, 
                                                "count":1,  # 这个时刻的统计信息出现了几次
                                                "max_depth": len(sc_nodes),     # 转发路径起点是源节点，长度可以用于计算级联以源节点为root节点的最长深度
                                                }
                else:
                    all_sub_state_dict[snapshot_time]['nodes'].update(set(sc_nodes))
                    all_sub_state_dict[snapshot_time]['edges'].update(sc_edges)
                    all_sub_state_dict[snapshot_time]['count'] += 1
                    all_sub_state_dict[snapshot_time]['max_depth'] = max( all_sub_state_dict[snapshot_time]['max_depth'], len(sc_nodes))
                # all_sub_state_dict[snapshot_time]['size'] = len(all_sub_state_dict[snapshot_time]['nodes']) - 1
            
            # 有边的信息就不需要再保存节点了
            # 因为级联中没有孤立节点，边信息就包含了所有节点
            # 但是边的信息可以通过遍历快照记录逐渐更新，不需要重复记录
            # 所以保存节点信息更方便
            cascade_data = {
                            # "graph_info":{
                            #                 "nodes":list(nodes_in_cascade),
                            #                 # "edges":list(edges_in_cascade),
                            #                 # "in_degree":in_degree,
                            #                 # "out_degree":out_degree
                            #               },
                            # "all_nodes":list(nodes_in_cascade),
                            "final_size":int(retweet_number),
                            "size_bias": total_size - int(retweet_number),
                            "publish_time":publish_time,
                            "time_for_sort":time_str_to_float_sort(publish_time),
                            "time_for_filter":time_str_to_int(publish_time, key_time_type),
                            "source_node":user_id,
                            "cascade_id":message_id,
                            "snapshot_times":list(sorted(all_sub_state_dict.keys())),
                            }
            new_edges = set()
            # current_nodes = set()
            current_size = 0        # 当前快照时刻的级联大小
            current_max_depth = 0
            keys = cascade_data['snapshot_times']
            for i, key in enumerate(keys):
                value = all_sub_state_dict[key]
                # current_nodes.update(value['nodes'])

                # 保存labels太耗内存了，动态生成吧
                # labels = { node:0 for node in nodes_in_cascade}
                # for n in value['nodes']:
                #     labels[n] = 1
                # cascade_data[f"graph_{i}"] = {"labels":labels, "occur_time":key}
                # cascade_data[f"graph_{i}"] = {"labels":labels, "occur_time":key, "edges":value["edges"]}

                # 保存每个时刻的边也有点浪费，只记录快照时刻新增加的边
                # cascade_data[f"graph_{i}"] = {"occur_time":key, "edges":value["edges"]}
                tmp_edges = value["edges"]
                # 保存节点也占内存
                # current_nodes.update(value['nodes'])
                current_size += value['count']
                current_max_depth = max( current_max_depth, value['max_depth'])
                cascade_data[f"snapshot_{i}"] = {"occur_time":key,  "activated_size": current_size,
                                                "new_edges": list(tmp_edges - new_edges),
                                                "max_depth": current_max_depth
                                                }
                # cascade_data[f"snapshot_{i}"] = {"occur_time":key,  "activated_size": current_size,
                #                                 "new_edges": list(tmp_edges - new_edges), "nodes":list(current_nodes)
                #                                 }
                new_edges.update(tmp_edges)
                
            all_cascade_data.append(cascade_data)
            # f_t.write(json.dumps(cascade_data) + '\n')
    
    # f_t.close()
    data_jsonl_saver(dataset_name, all_cascade_data, dataset_name)

def time_str_to_int(publish_time: str, key_time_type = 'h') -> int:
    '''
    str日期转为用于筛选时间范围的int
    '''
    # year-month-day
    if '-' in publish_time:
        return int(publish_time.split('-')[0])
    # timestamp to struct_time
    elif len(publish_time) == 10:
        # 0	tm_year	年, [1,9999]
        # 1	tm_mon	月, [1,12]
        # 2	tm_mday	日, [1,当月天数]
        # 3	tm_hour	时, [0,24)
        # 4	tm_min	分, [0,60)
        # 5	tm_sec	秒, [0,60)
        # 6	tm_wday	一周中的第几天, [0,6]
        # 7	tm_yday	一年中的第几天, [1,366?]
        # 8	tm_isdst	夏令时
        struct_time = time.localtime( float(publish_time))
        if key_time_type == 'h':
            return struct_time[3]
        elif key_time_type == 'd':
            return struct_time[1] * 100 + struct_time[2]
    # year
    return int(publish_time)

def time_str_to_float_sort(publish_time: str) -> float:
    '''
    str日期转为用于排序的float
    '''
    # year-month-day
    if '-' in publish_time:
        # win下不支持1970年以前的日期转化，改用日期拼接
        # return time.mktime(time.strptime(publish_time, "%Y-%m-%d"))
        return float(''.join(publish_time.split('-')))
    # timestamp to struct_time
    elif len(publish_time) == 10:
        return float(publish_time)
    # year
    # return time.mktime(time.strptime(publish_time, "%Y"))
    return float(publish_time)

def gen_snapshot_time_feature_by_timestamp(publish_time: str, snapshot_time: int or float) -> List[int]:
    '''
    publish_time: 源时间，字符串的形式可用于区分不同的时间跨度
    snapshot_time: 快照时间相距源时间的间隔，可能是不同时间单位
                    因此需要依靠publish_time的形式做区分
    '''
    # timestamp，对应推特和微博数据集
    if '-' not in publish_time and len(publish_time) == 10:
        # tmp_struct_time = time.gmtime(float(publish_time))
        tmp_datetime = datetime.datetime.fromtimestamp(float(publish_time))
        snapshot_datetime = tmp_datetime + datetime.timedelta(seconds=float(snapshot_time))
    # year-month-day  or  year， 对应论文数据集aps和dblp
    # win下不支持1970年以前的日期转化，需要先转接一下
    else:
        # year-month-day
        if '-' in publish_time:
            tmp_struct_time = time.strptime(publish_time, "%Y-%m-%d")
        # year
        else:
            tmp_struct_time = time.strptime(publish_time, "%Y")
        tmp_timestamp = calendar.timegm(tmp_struct_time)
        tmp_datetime = convert_timestamp_to_datetime(tmp_timestamp)

        snapshot_datetime = tmp_datetime + datetime.timedelta(days=int(snapshot_time))
    
    # 转为struct_time元组，方便提取信息，最后一位dst不需要
    snapshot_struct_time = snapshot_datetime.timetuple()[:8]
    return snapshot_struct_time

def convert_timestamp_to_datetime(timestamp):
    '''
    win下不支持1970年以前的日期转换 所以要绕一下
    根据时间戳生成对应时间对象
    '''
    if timestamp >= 0:
        return datetime.datetime.fromtimestamp(timestamp)
    else:
        return datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(timestamp)) 

# def merge_synthetic_data():
#     '''
#     仅使用一次，合并数据
#     '''
#     target_path = tmp_path / 'synthetic_data_V2'
#     target_path.mkdir(exist_ok=True)
#     source_path = dataset_path / 'synthetic_data_V2'
#     with open(target_path / 'synthetic_data_V2.jsonl', 'w', encoding='utf8') as f_t:
#         for ch in source_path.iterdir():
#             with open(ch) as f:
#                 tmp = json.load(f)
#             f_t.write( json.dumps(tmp) + '\n')


if __name__ == "__main__":

    # dataset_name = 'acm'
    # dataset_name = 'aps'
    # dataset_name = 'cascade_sample'
    # dataset_name = 'dblp'
    dataset_name = 'twitter'
    # dataset_name = 'weibo'
    # dataset_name = 'weibo_lite'
    transdata(dataset_name)
    # transdata(dataset_name, source_path=tmp_path / dataset_name / 'casflow_test.txt')


    datasets = []
    # datasets.append('acm')
    # datasets.append('aps')
    # datasets.append('cascade_sample')
    # datasets.append('dblp')
    # datasets.append('twitter')
    # datasets.append('weibo')
    for dataset_name in datasets:
        transdata(dataset_name)
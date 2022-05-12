import os, sys
import logging
import argparse
import time
import random
import traceback
from typing import List,Tuple
import json
import multiprocessing

import torch
import numpy as np
from sklearn.metrics import (mean_squared_log_error, mean_absolute_percentage_error,
                            mean_squared_error, mean_absolute_error, median_absolute_error,
                            r2_score, explained_variance_score)

from config import log_path, tmp_path, current_platform

train_logger = logging.getLogger('models')

def get_command_args():
    parser = argparse.ArgumentParser("CasInformer")
    # 训练参数
    parser.add_argument('-d', '--dataset', type=str, help='select dataset name',
            choices=['acm', 'aps', 'twitter', 'weibo', 'weibo_lite', 'cascade_sample', 'dblp'],)
    parser.add_argument('-lr', '--learning_rate', type=float, help='default 0.001')
    parser.add_argument('--weight_decay', type=float, help='default 0.001')
    parser.add_argument('--optimizer_type', type=str, help='default Adam', choices=['Adam', 'AdamW'])
    parser.add_argument('--optimizer_eps', type=float, help='default 1e-9')
    parser.add_argument('--optimizer_beta1', type=float, help='default 0.9')
    parser.add_argument('--optimizer_beta2', type=float, help='default 0.999')
    parser.add_argument('--optimizer_clip_value', type=float, help='default 10')
    parser.add_argument('--loss_type', type=str, help='default msle', 
                            choices=['msle', 'msle_log2', 'msle_log10', 'mae','mse','huber','smooth_l1'])

    parser.add_argument('--epochs', type=int, help='default 1000')
    parser.add_argument('--check_point', type=int, help='default 1')
    early_stop = parser.add_mutually_exclusive_group()
    early_stop.add_argument('--early_stop', default=None, help='default early_stop', action="store_true")
    early_stop.add_argument('--not_early_stop', default=None, help='default early_stop', action="store_false")
    parser.add_argument('-es','--early_stop_patience', type=int, help='default 6')
    parser.add_argument('--early_stop_min_delta', type=float, help='default 0')

    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument('--warmup', default=None, help='default warmup', action="store_true")
    warmup.add_argument('--not_warmup', default=None, help='default warmup', action="store_false")
    parser.add_argument('--warmup_steps', type=int, help='default 500')
    parser.add_argument('--warmup_times', type=int, help='default 2')
    parser.add_argument('--init_lr', type=float, help='default 3e-5')

    reduce_lr = parser.add_mutually_exclusive_group()
    reduce_lr.add_argument('--reduce_lr', default=None, help='default reduce_lr', action="store_true")
    reduce_lr.add_argument('--not_reduce_lr', default=None, help='default reduce_lr', action="store_false")
    parser.add_argument('--reduce_lr_patience', type=int, help='default 2')
    parser.add_argument('--reduce_lr_factor', type=float, help='default 0.1')
    parser.add_argument('--cool_down', type=int, help='default 0')
    parser.add_argument('--min_lr', type=float, help='default 1e-7')
    parser.add_argument('--reduce_lr_eps', type=float, help='default 1e-9')
    
    parser.add_argument('-bs1', '--train_batch_size', type=int, help='default 32, depends on datasets')
    parser.add_argument('-bs2', '--val_batch_size', type=int, help='default 32, depends on datasets')
    parser.add_argument('-bs3', '--test_batch_size', type=int, help='default 32, depends on datasets')
    parser.add_argument('--device', type=str, help='default device', default=None, choices=['cpu', '0','1','2','3'])
    use_multi_gpu = parser.add_mutually_exclusive_group()
    use_multi_gpu.add_argument('--use_multi_gpu', default=None, help='default early_stop', action="store_true")
    use_multi_gpu.add_argument('--not_use_multi_gpu', default=None, help='default early_stop', action="store_false")
    parser.add_argument('--gpu_ids', type=str, help='default use 0,1')
    use_amp = parser.add_mutually_exclusive_group()
    use_amp.add_argument('--use_amp', default=None, help='default True', action="store_true")
    use_amp.add_argument('--not_use_amp', default=None, help='default False', action="store_false")
    pin_memory = parser.add_mutually_exclusive_group()
    pin_memory.add_argument('--pin_memory', default=None, help='default True', action="store_true")
    pin_memory.add_argument('--not_pin_memory', default=None, help='default False', action="store_false")
    pre_load = parser.add_mutually_exclusive_group()
    pre_load.add_argument('--pre_load', default=None, help='default True', action="store_true")
    pre_load.add_argument('--not_pre_load', default=None, help='default False', action="store_false")
    drop_last = parser.add_mutually_exclusive_group()
    drop_last.add_argument('--drop_last', default=None, help='default True', action="store_true")
    drop_last.add_argument('--not_drop_last', default=None, help='default False', action="store_false")
    parser.add_argument('--loader', type=str, help='default batch')
    clean_cache = parser.add_mutually_exclusive_group()
    clean_cache.add_argument('--clean_cache', default=None, help='default True', action="store_true")
    clean_cache.add_argument('--not_clean_cache', default=None, help='default False', action="store_false")
    average_batch_data_size = parser.add_mutually_exclusive_group()
    average_batch_data_size.add_argument('--average_batch_data_size', default=None, help='default True', action="store_true")
    average_batch_data_size.add_argument('--not_average_batch_data_size', default=None, help='default False', action="store_false")
    
    parser.add_argument('--random_seed', type=int, help='default 3')
    stable = parser.add_mutually_exclusive_group()
    stable.add_argument('--stable', default=None, help='default True', action="store_true")
    stable.add_argument('--not_stable', default=None, help='default False', action="store_false")

    parser.add_argument('-goal','--predict_type', type=str, help='default full', choices=['full', 'gap'])

    parser.add_argument('--run_all', help='run all datasets', action="store_true")

    # 模型参数
    parser.add_argument('--hidden_size', type=int, help='default 32')

    parser.add_argument('--gnn_type', type=str, help='default GCN2', choices=['GCN', 'GCN2', 'GAT', 'GAT2'])
    parser.add_argument('--gnn_hidden_channels', type=int, help='default 0.001')
    parser.add_argument('--gnn_num_layers', type=int, help='default 3')
    parser.add_argument('--gnn_out_channels', type=int, help='default 32')
    parser.add_argument('--gnn_dropout', type=float, help='default 0.1')
    parser.add_argument('--dynamic_route_iter', type=int, help='default 3')
    parser.add_argument('--capsule_out_dim', type=int, help='default 64')
    parser.add_argument('-r','--rnn_type', type=str, help='default GRU', choices=['LSTM', 'GRU', 'Informer'])
    parser.add_argument('--rnn_num_layers', type=int, help='default 2, valid only for LSTM or GRU')
    parser.add_argument('--rnn_dropout', type=float, help='default 0.1')
    parser.add_argument('--rnn_hidden_size', type=int, help='default 64')
    rnn_bidirectional = parser.add_mutually_exclusive_group()
    rnn_bidirectional.add_argument('--rnn_bidirectional', help='default True, valid only for LSTM or GRU', default=None, action="store_true")
    rnn_bidirectional.add_argument('--not_rnn_bidirectional', help='default False', default=None, action="store_false")
    parser.add_argument('--mlp_hidden_size', type=int, help='default 64')
    parser.add_argument('--mlp_dropout', type=float, help='default 0.5')
    parser.add_argument('--mlp_outsize', type=int, help='default 1')
    parser.add_argument('--mlp_num_layers', type=int, help='default 3')

    # 数据设置
    parser.add_argument('--dim_of_state_features', type=int, help='default 2')
    node_state = parser.add_mutually_exclusive_group()
    node_state.add_argument('--use_node_state_feature', default=None, help='default True', action="store_true")
    node_state.add_argument('--not_use_node_state_feature', default=None, help='default False', action="store_false")
    parser.add_argument('--dim_of_degree_features', type=int, help='default 2')
    node_degree = parser.add_mutually_exclusive_group()
    node_degree.add_argument('--use_node_degree_feature', default=None, help='default True', action="store_true")
    node_degree.add_argument('--not_use_node_degree_feature', default=None, help='default False', action="store_false")
    parser.add_argument('--dim_of_time_features', type=int, help='default 2')
    node_time = parser.add_mutually_exclusive_group()
    node_time.add_argument('--use_node_time_feature', default=None, help='default True', action="store_true")
    node_time.add_argument('--not_use_node_time_feature', default=None, help='default False', action="store_false")
    parser.add_argument('--dim_of_depth_features', type=int, help='default 1')
    node_depth = parser.add_mutually_exclusive_group()
    node_depth.add_argument('--use_node_depth_feature', default=None, help='default True', action="store_true")
    node_depth.add_argument('--not_use_node_depth_feature', default=None, help='default False', action="store_false")
    parser.add_argument('--dim_of_cascade_depth', type=int, help='default 1')
    cascade_depth = parser.add_mutually_exclusive_group()
    cascade_depth.add_argument('--use_cascade_depth', default=None, help='default True', action="store_true")
    cascade_depth.add_argument('--not_use_cascade_depth', default=None, help='default False', action="store_false")
    parser.add_argument('--dim_of_cascade_activated_size', type=int, help='default 1')
    cascade_size = parser.add_mutually_exclusive_group()
    cascade_size.add_argument('--use_cascade_activated_size', default=None, help='default True', action="store_true")
    cascade_size.add_argument('--not_use_cascade_activated_size', default=None, help='default False', action="store_false")
    parser.add_argument('--dim_of_cascade_time', type=int, help='default 1')
    cascade_time = parser.add_mutually_exclusive_group()
    cascade_time.add_argument('--use_cascade_time', default=None, help='default True', action="store_true")
    cascade_time.add_argument('--not_use_cascade_time', default=None, help='default False', action="store_false")
    parser.add_argument('--cascade_level_features_dim',type=int, help='default 16')
    cascade_feature = parser.add_mutually_exclusive_group()
    cascade_feature.add_argument('--use_cascade_level_features_in_rnn', default=None, help='default True', action="store_true")
    cascade_feature.add_argument('--not_use_cascade_level_features_in_rnn', default=None, help='default False', action="store_false")
    parser.add_argument('--time_tuple_features_dim', type=int, help='default 7, depends on datasets')

    parser.add_argument('-ns','--number_of_snapshot', type=int, help='default 12, depends on datasets and observe time')

    parser.add_argument('-o','--observe_time_unit_num', type=float, help='default 1, observe 1 time unit, depends on datasets')
    parser.add_argument('-p','--predict_time_unit_num', type=float, help='default 24(weibo), predict 24 time unit, depends on datasets')
    parser.add_argument('--time_coefficient', type=int, help='default 1, depends on datasets')
    parser.add_argument('-min','--min_size_in_observe_window', type=int, help='default 0, means no limit, depends on datasets and observe time')
    parser.add_argument('-max','--max_size_in_observe_window', type=int, help='default 0, means no limit, depends on datasets and observe time')
    parser.add_argument('-start','--publish_start', type=int, help='default 0, means no limit, depends on datasets')
    parser.add_argument('-end','--publish_end', type=int, help='default 0, means no limit, depends on datasets')
    
    parser.add_argument('--train_ratio', type=float, help='default 0.7, depends on datasets')
    parser.add_argument('--val_ratio', type=float, help='default 0.15, depends on datasets')
    parser.add_argument('--test_ratio', type=float, help='default 0.15, depends on datasets')
    random_split = parser.add_mutually_exclusive_group()
    random_split.add_argument('--random_split', default=None, help='default True', action="store_true")
    random_split.add_argument('--not_random_split', default=None, help='default False', action="store_false")
    sort_by_time = parser.add_mutually_exclusive_group()
    sort_by_time.add_argument('--sort_by_time', default=None, help='default True', action="store_true")
    sort_by_time.add_argument('--not_sort_by_time', default=None, help='default False', action="store_false")

    args = parser.parse_args()
    # print(args)
    return args

def adjust_config_by_dataset(dataset_name, default_config) -> dict:
    '''
    根据预设的数据集对应配置调整config
    '''
    print(f'adjust config for dataset {dataset_name}')
    if dataset_name == 'weibo_lite':
        dataset_name = 'weibo'
    if dataset_name in default_config['dataset_custom_settings'].keys():
        for key,value in default_config['dataset_custom_settings'][dataset_name]['model'].items():
            default_config['model_settings'][key] = value
        for key,value in default_config['dataset_custom_settings'][dataset_name]['data'].items():
            default_config['data_settings'][key] = value
        for key,value in default_config['dataset_custom_settings'][dataset_name]['train'].items():
            default_config['train_settings'][key] = value
    else:
        print(f'WARNING! no config for this dataset {dataset_name}, use default config')
    
    return default_config

def adjust_config_auto(default_config):
    '''
    根据输入参数调整config
    '''
    # node level
    default_config['data_settings']['dim_of_node_features'] = 0
    default_config['data_settings']['dim_of_node_features'] += (default_config['data_settings']['dim_of_state_features'] 
                                                                if default_config['data_settings']['use_node_state_feature'] else 0)
    default_config['data_settings']['dim_of_node_features'] += (default_config['data_settings']['dim_of_degree_features'] 
                                                                if default_config['data_settings']['use_node_degree_feature'] else 0)
    default_config['data_settings']['dim_of_node_features'] += (default_config['data_settings']['dim_of_time_features'] 
                                                                if default_config['data_settings']['use_node_time_feature'] else 0)
    default_config['data_settings']['dim_of_node_features'] += (default_config['data_settings']['dim_of_depth_features'] 
                                                                if default_config['data_settings']['use_node_depth_feature'] else 0)

    # cascade level
    default_config['data_settings']['dim_of_cascade_features'] = 0
    default_config['data_settings']['dim_of_cascade_features'] += (default_config['data_settings']['dim_of_cascade_activated_size'] 
                                                                if default_config['data_settings']['use_cascade_activated_size'] else 0)
    default_config['data_settings']['dim_of_cascade_features'] += (default_config['data_settings']['dim_of_cascade_depth'] 
                                                                if default_config['data_settings']['use_cascade_depth'] else 0)
    default_config['data_settings']['dim_of_cascade_features'] += (default_config['data_settings']['dim_of_cascade_time'] 
                                                                if default_config['data_settings']['use_cascade_time'] else 0)
    # 调整网络隐层维度
    

    # 根据参数设置观察时间
    assert default_config['data_settings']['observe_time'] != 0, 'observe time can not be 0'
    assert default_config['data_settings']['predict_time'] != 0, 'predict time can not be 0'
    default_config['data_settings']['observe_time'] = default_config['data_settings']['observe_time_unit_num'] * default_config['data_settings']['time_per_unit']
    if default_config['data_settings']['unit_bias']:
        default_config['data_settings']['observe_time'] += default_config['data_settings']['observe_time_unit_num'] // default_config['data_settings']['unit_bias']
    
    default_config['data_settings']['predict_time'] = default_config['data_settings']['predict_time_unit_num'] * default_config['data_settings']['time_per_unit']
    if default_config['data_settings']['unit_bias']:
        default_config['data_settings']['predict_time'] += default_config['data_settings']['predict_time_unit_num'] // default_config['data_settings']['unit_bias']
    
    # 设置Informer的预测序列长度
    expect_length = default_config['data_settings']['number_of_snapshot']
    label_part_length = default_config['data_settings']['number_of_snapshot'] // 2
    predict_part_length =  expect_length - label_part_length
    default_config['data_settings']['expect_length'] = expect_length
    default_config['data_settings']['label_part_length'] = label_part_length
    default_config['data_settings']['predict_part_length'] = predict_part_length

    # rnn始终无法稳定复现，干脆不设置了
    # if default_config['model_settings']['rnn_type'] != 'Informer':
    #     default_config['train_settings']['stable'] = False
    
    return default_config

def adjust_config_by_args(args, default_config, dataset_name):
    '''
    从命令行arg中更新config
    '''
    if args.dataset:
        dataset_name = args.dataset
    
    default_config = adjust_config_by_dataset(dataset_name, default_config)

    if args.rnn_bidirectional:
        default_config['model_settings']['rnn_bidirectional'] = True
    if args.not_rnn_bidirectional == False:
        default_config['model_settings']['rnn_bidirectional'] = False
    
    if args.use_node_state_feature:
        default_config['data_settings']['use_node_state_feature'] = True
    if args.not_use_node_state_feature == False:
        default_config['data_settings']['use_node_state_feature'] = False
    if args.use_node_degree_feature:
        default_config['data_settings']['use_node_degree_feature'] = True
    if args.not_use_node_degree_feature == False:
        default_config['data_settings']['use_node_degree_feature'] = False
    if args.use_node_time_feature:
        default_config['data_settings']['use_node_time_feature'] = True
    if args.not_use_node_time_feature == False:
        default_config['data_settings']['use_node_time_feature'] = False
    if args.use_node_depth_feature:
        default_config['data_settings']['use_node_depth_feature'] = True
    if args.not_use_node_depth_feature == False:
        default_config['data_settings']['use_node_depth_feature'] = False
    if args.use_cascade_depth:
        default_config['data_settings']['use_cascade_depth'] = True
    if args.not_use_cascade_depth == False:
        default_config['data_settings']['use_cascade_depth'] = False
    if args.use_cascade_activated_size:
        default_config['data_settings']['use_cascade_activated_size'] = True
    if args.not_use_cascade_activated_size == False:
        default_config['data_settings']['use_cascade_activated_size'] = False
    if args.use_cascade_time:
        default_config['data_settings']['use_cascade_time'] = True
    if args.not_use_cascade_time == False:
        default_config['data_settings']['use_cascade_time'] = False
    if args.use_cascade_level_features_in_rnn:
        default_config['data_settings']['use_cascade_level_features_in_rnn'] = True
    if args.not_use_cascade_level_features_in_rnn == False:
        default_config['data_settings']['use_cascade_level_features_in_rnn'] = False
    
    if args.random_split:
        default_config['train_settings']['random_split'] = True
    if args.not_random_split == False:
        default_config['train_settings']['random_split'] = False
    if args.sort_by_time:
        default_config['data_settings']['sort_by_time'] = True
    if args.not_sort_by_time == False:
        default_config['data_settings']['sort_by_time'] = False
    
    if args.early_stop:
        default_config['train_settings']['early_stop'] = True
    if args.not_early_stop == False:
        default_config['train_settings']['early_stop'] = False
    if args.reduce_lr:
        default_config['train_settings']['reduce_lr'] = True
    if args.not_reduce_lr == False:
        default_config['train_settings']['reduce_lr'] = False
    if args.warmup:
        default_config['train_settings']['warmup'] = True
    if args.not_warmup == False:
        default_config['train_settings']['warmup'] = False
    if args.pin_memory:
        default_config['train_settings']['pin_memory'] = True
    if args.not_pin_memory == False:
        default_config['train_settings']['pin_memory'] = False
    if args.pre_load:
        default_config['train_settings']['pre_load'] = True
    if args.not_pre_load == False:
        default_config['train_settings']['pre_load'] = False
    if args.drop_last:
        default_config['train_settings']['drop_last'] = True
    if args.not_drop_last == False:
        default_config['train_settings']['drop_last'] = False
    if args.use_multi_gpu:
        default_config['train_settings']['use_multi_gpu'] = True
    if args.not_use_multi_gpu == False:
        default_config['train_settings']['use_multi_gpu'] = False
    if args.use_amp:
        default_config['train_settings']['use_amp'] = True
    if args.not_use_amp == False:
        default_config['train_settings']['use_amp'] = False
    if args.clean_cache:
        default_config['train_settings']['clean_cache'] = True
    if args.not_clean_cache == False:
        default_config['train_settings']['clean_cache'] = False
    if args.average_batch_data_size:
        default_config['train_settings']['average_batch_data_size'] = True
    if args.not_average_batch_data_size == False:
        default_config['train_settings']['average_batch_data_size'] = False
    if args.stable:
        default_config['train_settings']['stable'] = True
    if args.not_stable == False:
        default_config['train_settings']['stable'] = False
    
    arg_dict = vars(args)
    for k,v in arg_dict.items():
        if not isinstance(v, bool) and (v != None) and k in default_config['model_settings']:
            default_config['model_settings'][k] = v
        if not isinstance(v, bool) and (v != None) and k in default_config['data_settings']:
            default_config['data_settings'][k] = v
        if not isinstance(v, bool) and (v != None) and k in default_config['train_settings']:
            default_config['train_settings'][k] = v
    
    return dataset_name, default_config


def set_device(default_config):
    # Set GPU
    # if not current_platform.startswith('win'):
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"      # 服务器上多显卡选择第二张显卡使用

    if default_config['train_settings']['device'] == 'cpu':
        print('device set to cpu, use cpu ')
        device = torch.device('cpu')
        default_config['train_settings']['use_amp'] = False

    elif torch.cuda.is_available():
        print('GPU now available ', end = ', ')
        number_of_gpu = torch.cuda.device_count()
        if default_config['train_settings']['use_multi_gpu'] and number_of_gpu > 1:
            print(f"{number_of_gpu} gpu available, use {default_config['train_settings']['gpu_ids']}")
            # 设置可见显卡
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = default_config['train_settings']['gpu_ids']
            
            # default_config['train_settings']['gpu_ids'] = [i for i,id in enumerate(default_config['train_settings']['gpu_ids'].split(',')) ]
            default_config['train_settings']['pin_memory'] = False
            device = torch.device(f"cuda:0")    # 主GPU
        else:
            print(f"{number_of_gpu} gpu available, use_multi_gpu is {default_config['train_settings']['use_multi_gpu']}, use single gpu {default_config['train_settings']['device']}")
            default_config['train_settings']['use_multi_gpu'] = False
            
            device = torch.device(f"cuda:{default_config['train_settings']['device']}")
        # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        print('no gpu available, use cpu ')
        device = torch.device('cpu')
        default_config['train_settings']['use_amp'] = False
    
    return device

def setup_seed(seed):
    '''
    固定随机数种子
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'            # cuda 10.1
        # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'     # cuda 10.2+, 4MB, rnn可复现性设置，实测不管怎么设置都不稳定
        # os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'       # 16KB, , rnn可复现性设置 和cuda可复现设置
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'     # 24MB, cuda可复现设置, 没有提到rnn，informer可稳定复现
        torch.set_deterministic_debug_mode('warn')          # 文档中写可替代use_deterministic_algorithms，
                                                            # 但实测并没有生效，只有对应的提示
                                                            # warn(str) or 1(int) , 
                                                            # error(str) or 2(int), 
                                                            # default(str) or 0(int)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

class TrainHelper():
    """
    Early stopping, Warm Up, Reduce LR
    """
    def __init__(self, optimizer, 
                warmup: bool = True, warmup_steps: int = 1000, warmup_times: int = 2, train_steps: int = 1000,
                init_lr = 1e-7, max_lr = 1e-3,
                reduce_lr: bool = True, reduce_lr_patience = 2, reduce_lr_factor = 0.1, cool_down = 0,
                min_lr = 1e-7, eps = 1e-9,
                early_stop: bool = True, early_stop_patience: int = 5, min_delta: float = 0.,
                ):
        """
        
        """
        self.optimizer = optimizer

        self.warmup = warmup
        self.warmup_steps = warmup_steps
        self.warmup_complete = False
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.current_steps = 0
        if self.warmup:
            if self.warmup_steps == 0:
                self.warmup_steps = warmup_times * train_steps
            self.lr_increasement = (max_lr - init_lr) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = init_lr

        self.reduce_lr = reduce_lr
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_counter = 0
        self.reduced = False
        self.cool_down = cool_down
        self.cool_down_counter = 0
        self.min_lr = min_lr
        self.eps = eps

        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.need_stop = False
        self.early_stop_counter = 0
        self.best_loss = None
        
    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()
        self.current_steps += 1
        # print(self.current_steps)
        if self.warmup and (self.current_steps <= self.warmup_steps):
            new_lr = self.init_lr + self.lr_increasement * self.current_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            self.warmup_complete = True
            # print(f'set warmup complete flag {self.warmup_complete}')
    
    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def __call__(self, val_loss) -> bool:
        if val_loss != val_loss:
            print('WARNING: Loss is NaN, stop training')
            self.need_stop = True
            return False
        
        print(f'this time val loss is {val_loss}, best loss before is {self.best_loss}')

        if self.best_loss == None:
            self.best_loss = val_loss
        
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.early_stop_counter = 0
            self.reduce_lr_counter = 0
        
        elif self.best_loss - val_loss < self.min_delta:
            self.early_stop_counter += 1
            if self.reduced:
                self.cool_down_counter += 1
                if self.cool_down_counter > self.cool_down:
                    self.reduce_lr_counter += 1
            else:
                self.reduce_lr_counter += 1
            
            if self.reduce_lr:
                print(f"INFO: Reduce lr counter {self.reduce_lr_counter} of {self.reduce_lr_patience}")
            if self.early_stop:
                print(f"INFO: Early stopping counter {self.early_stop_counter} of {self.early_stop_patience}")

            # print(self.warmup_complete)
            if self.reduce_lr and self.warmup_complete and self.reduce_lr_counter >= self.reduce_lr_patience:
                for param_group in self.optimizer.param_groups:
                    new_lr = max( param_group['lr'] * self.reduce_lr_factor, self.min_lr)
                    if param_group['lr'] - new_lr > self.eps:
                        param_group['lr'] = new_lr
                self.reduce_lr_counter = 0
                self.cool_down_counter = 0
                self.reduced = True
                print(f"reduce lr to {new_lr}")
            
            if self.early_stop and self.early_stop_counter >= self.early_stop_patience:
                print('INFO: Early stopping')
                self.need_stop = True
            
            return False
        
        return True


def prediction_metrics(pred, y, full_pred, full_y):
    '''
    pred: 增量
    y:  增量
    full_pred: 完整规模，用于计算MRSE和WroPerc等
    full_y: 完整规模
    '''
    # eps = 1e-5
    # train_logger.debug(pred)
    # train_logger.debug(y)
    pred = np.maximum(pred, 0)
    y = np.maximum(y, 0)
    # train_logger.debug(pred)
    # train_logger.debug(y)
    metric_methods_1 = {
        'MSLE_log2':lambda y, pred:np.mean(np.square(np.log2(pred + 1) - np.log2(y + 1))),
        'MAPE_log2':lambda y, pred:np.mean(np.abs(np.log2(pred + 1) - np.log2(y + 1)) / np.log2(y + 2)),
    }
    metric_methods_2 = {
        'MSLE':mean_squared_log_error,
        'MAPE':mean_absolute_percentage_error,
        'MRSE':lambda y,pred: np.mean(np.square( (pred - y) / (y) )),         # coupled gnn
        'WroPerc':lambda y,pred: np.sum(np.abs(pred - y) / (y)  > 0.5) / y.size, # coupled gnn
        'MSE':mean_squared_error,
        'RMSE':lambda y, pred: mean_squared_error(y, pred, squared=False),
        'MAE':mean_absolute_error,
        'mAE':median_absolute_error,
        # 'R2':r2_score,
        # 'EVS':explained_variance_score,
    }
    results = {}
    for key, func in metric_methods_1.items():
        try:
            results[key] = float(func(y, pred))
        except ValueError:
            # traceback.print_exc()
            train_logger.warn(f'ValueError in calculate metric {key}')
            train_logger.warn(pred)
            train_logger.warn(y)
    
    for key, func in metric_methods_2.items():
        try:
            results[key] = float(func(full_y, full_pred))
        except ValueError:
            # traceback.print_exc()
            train_logger.warn(f'ValueError in calculate metric {key}')
            train_logger.warn(pred)
            train_logger.warn(y)
    
    # results['mSLE']
    # results['RMSPE']
    # results['mRSE']

    return results

def trans_time_feature(struct_time_feature: List[Tuple[int]], end: int = 7) -> List[float]:
    '''
    时间特征初始化编码
    '''
    index_function_dict = {
        1: month_in_year_trans,
        2: day_in_month_trans,
        3: hour_in_day_trans,
        4: minute_in_hour_trans,
        5: second_in_mintue_trans,
        6: day_in_week_trans,
        7: day_in_year_trans,
    }

    time_feature = []
    for i, tf in enumerate(struct_time_feature):
        if i == 0:
            continue
        if i > end:
            break
        time_feature.append(index_function_dict[i](tf))
    
    return time_feature

def second_in_mintue_trans(second: int) -> float:
    return (float(second) / 59.0) - 0.5

def minute_in_hour_trans(minute: int) -> float:
    return (float(minute) / 59.0) - 0.5

def hour_in_day_trans(hour: int) -> float:
    return (float(hour) / 23.0) - 0.5

def day_in_month_trans(day: int) -> float:
    return (float(day - 1) / 31.0) - 0.5

def month_in_year_trans(month: int) -> float:
    return (float(month - 1) / 11.0) - 0.5

def day_in_week_trans(weekday: int) -> float:
    return (float(weekday) / 6.0) - 0.5

def day_in_year_trans(day: int) -> float:
    return (float(day - 1) / 365.0) - 0.5

# def year_in_history_trans(year: int) -> float:
#     return (float(year - 2000) / 100.0)


def data_jsonl_loader(dataset_name: str, target_file_name: str) -> List[dict]:
    '''
    读取jsonl文件
    '''
    all_item = []
    if target_file_name.endswith('.jsonl'):
        target_file_name = target_file_name.rstrip('.jsonl')
    with open(tmp_path / dataset_name / f'{target_file_name}.jsonl', 'rb') as f:
        for line in f.readlines():
            all_item.append(json.loads(line))
    return all_item

def data_jsonl_saver(dataset_name: str, all_item: List[dict] , target_file_name: str):
    '''
    保存jsonl文件
    '''
    (tmp_path / dataset_name).mkdir(exist_ok=True)
    with open(tmp_path / dataset_name / f'{target_file_name}.jsonl', 'w', encoding='utf8') as f:
        for item in all_item:
            f.write( json.dumps(item) + '\n')

def get_process_num() -> int:
    '''
    获取cpu逻辑核心数
    '''
    core_num = multiprocessing.cpu_count()
    if core_num >= 6:
        process_num = min(core_num -2, 8)
    elif core_num > 2:
        process_num = core_num -1
    else:
        process_num = core_num
    print(f'system have  {core_num} threadings, create {process_num} process')
    return process_num

class Logger(object):
    '''
    自定义一个logger，用于将print输出同时保存到文件中
    使用方法
    sys.stdout = Logger("log_filename")
    '''
    def __init__(self, fileN="output.log", write_mode = 'w'):
        self.terminal = sys.stdout
        self.log = open(fileN, write_mode, encoding='utf-8')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() #每次写入后刷新到文件中，防止程序意外结束
        
    def flush(self):
        self.log.flush()


if __name__ == "__main__":
    pass
    # er=nx.erdos_renyi_graph(5,0.6)
    # print(er.nodes())
    # embeddings = gen_node_embedding(er, embedding_method_name='graphwave')
    # embeddings = gen_node_embedding(er, embedding_method_name='role2vec')
    # embeddings = gen_node_embedding(er, embedding_method_name='prone')
    # print(len(embeddings))
    # print(len(embeddings[0]))
import sys
import time
import traceback
import logging
import pprint
import json
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Subset
from torch.utils.data import DataLoader as DataLoaderPytroch
from torch_geometric.loader import DataLoader as DataLoaderPyg
from torch_geometric.loader import DataListLoader as DataListLoaderPyg
from torch_geometric.nn.data_parallel import DataParallel as DataParallelPyg

from tqdm import tqdm
import numpy as np

from config import get_icp_default_config, current_platform, model_path, result_path, log_path
from load_data import CasInformerDataListPyG, CasInformerDataSetPyG
from models import CasInformer
from train_utils import (get_command_args, adjust_config_by_args, adjust_config_auto,
                    set_device, setup_seed, TrainHelper, prediction_metrics)
from pub_utils import set_logging, Logger

# debug
# torch.autograd.set_detect_anomaly(True)

logging.disable(logging.DEBUG)
train_logger = logging.getLogger('models')


def train_and_eval(dataset_name: str, default_config: dict, current_time):
    '''
    
    '''
    device = set_device(default_config)
    # print(f'current available device: {device}')

    # 随机生成并固定随机数
    if default_config['train_settings']['stable']:
        # default_config['model_settings']['random_seed'] = int(time.time() * 256)
        setup_seed(default_config['train_settings']['random_seed'])

    pprint.pprint(default_config['model_settings'], sort_dicts=False)
    pprint.pprint(default_config['train_settings'], sort_dicts=False)
    pprint.pprint(default_config['data_settings'], sort_dicts=False)

    # 测试时调整加载数据的方法
    # load_data_method = 'pyg'

    train_logger.debug('begin, load dataset, GPU memory ')
    train_logger.debug(torch.cuda.memory_allocated(device))
    train_logger.debug(torch.cuda.memory_reserved(device))
    train_logger.debug(torch.cuda.max_memory_allocated(device))
    
    # 加载数据
    train_loader, val_loader, test_loader = dataset_to_loader(dataset_name, default_config)
    train_steps = len(train_loader)
    
    train_logger.debug('load dataset down, init model, GPU memory ')
    train_logger.debug(torch.cuda.memory_allocated(device))
    train_logger.debug(torch.cuda.memory_reserved(device))
    train_logger.debug(torch.cuda.max_memory_allocated(device))
    
    # 初始化模型
    print('init model...', end=', ')
    model = CasInformer(default_config['model_settings'], 
                    default_config['data_settings'], 
                    default_config['train_settings'],
                ).to(device)
    if default_config['train_settings']['use_multi_gpu']:
        model = DataParallelPyg(model, 
                device_ids = [i for i,id in enumerate(default_config['train_settings']['gpu_ids'].split(',')) ] )
    # else:
    model.train()
    # Adam or AdamW
    optim_dict = {
        "Adam":torch.optim.Adam,
        "AdamW":torch.optim.AdamW
    }
    optimizer = optim_dict[default_config['train_settings']['optimizer_type']](model.parameters(),
                                          lr=default_config['train_settings']['learning_rate'],
                                          weight_decay=default_config['train_settings']['weight_decay'],
                                          eps=default_config['train_settings']['optimizer_eps'],
                                          betas=[default_config['train_settings']['optimizer_beta1'], 
                                                default_config['train_settings']['optimizer_beta2']],
                                          )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=1,verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5, T_mult=2)
    if default_config['train_settings']['use_amp']:
        amp_scaler = torch.cuda.amp.GradScaler()
    train_helper = TrainHelper( optimizer = optimizer,
                                warmup = default_config['train_settings']['warmup'],
                                warmup_steps = default_config['train_settings']['warmup_steps'],
                                warmup_times = default_config['train_settings']['warmup_times'],
                                train_steps = train_steps,
                                init_lr = default_config['train_settings']['init_lr'],
                                max_lr = default_config['train_settings']['learning_rate'],

                                reduce_lr = default_config['train_settings']['reduce_lr'],
                                reduce_lr_patience = default_config['train_settings']['reduce_lr_patience'],
                                reduce_lr_factor = default_config['train_settings']['reduce_lr_factor'],
                                cool_down = default_config['train_settings']['cool_down'],
                                min_lr = default_config['train_settings']['min_lr'],
                                eps = default_config['train_settings']['reduce_lr_eps'],

                                early_stop = default_config['train_settings']['early_stop'],
                                early_stop_patience = default_config['train_settings']['early_stop_patience'],
                                min_delta = default_config['train_settings']['early_stop_min_delta'],
                                )
    
    train_logger.debug('init down, start training, GPU memory ')
    train_logger.debug(torch.cuda.memory_allocated(device))
    train_logger.debug(torch.cuda.memory_reserved(device))
    train_logger.debug(torch.cuda.max_memory_allocated(device))

    # 训练模型
    print('start training model...')
    # best_loss = float('inf')
    checkpoint_path = model_path / 'checkpoint'
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    model_save_path = str(checkpoint_path / f'CasInformer_{dataset_name}_{current_platform}_{current_time}.pt')
    for epoch in range(default_config['train_settings']['epochs']):
        train_logger.debug('start new epoch, GPU memory ')
        train_logger.debug(torch.cuda.memory_allocated(device))
        train_logger.debug(torch.cuda.memory_reserved(device))
        train_logger.debug(torch.cuda.max_memory_allocated(device))

        epoch_loss = 0.
        for batch in tqdm(train_loader):
            # try:
            train_helper.zero_grad()

            train_logger.debug('train one batch, GPU memory ')
            train_logger.debug(torch.cuda.memory_allocated(device))
            train_logger.debug(torch.cuda.memory_reserved(device))
            train_logger.debug(torch.cuda.max_memory_allocated(device))

            batch_loss = in_out_pyg_batch(model, batch, device = device, default_config = default_config, eval = False)
            
            train_logger.debug('train one batch down, backward, GPU memory ')
            train_logger.debug(torch.cuda.memory_allocated(device))
            train_logger.debug(torch.cuda.memory_reserved(device))
            train_logger.debug(torch.cuda.max_memory_allocated(device))

            if default_config['train_settings']['use_amp']:
                amp_scaler.scale(batch_loss).backward()
                amp_scaler.step(train_helper.optimizer)
                amp_scaler.update()
            else:
                batch_loss.backward()
                train_helper.step()
            
            
            # nn.utils.clip_grad_value_(model.parameters(), default_config['train_settings']['optimizer_clip_value'])
            # 遍历模型参数debug很耗时，不需要时注释掉
            # for name, params in model.named_parameters():
            #     train_logger.debug(f'name {name}')
            #     train_logger.debug(f'-->grad_requirs {params.requires_grad}')
            #     train_logger.debug(f'-->grad_value {params.grad}')
            epoch_loss += batch_loss.item()
            # del batch_loss
            train_logger.debug('backward down, GPU memory ')
            train_logger.debug(torch.cuda.memory_allocated(device))
            train_logger.debug(torch.cuda.memory_reserved(device))
            train_logger.debug(torch.cuda.max_memory_allocated(device))

            if default_config['train_settings']['clean_cache']:
                torch.cuda.empty_cache()

            # except RuntimeError as e:
            #     pass


        # 输出本轮loss
        epoch_loss = epoch_loss / train_steps
        print(f'train loss in epoch {epoch} is {epoch_loss}')
        # end_time = time.time()
        # print(end_time - start_time)

        # 定期验证数据并保存模型
        if (epoch + 1) % default_config['train_settings']['check_point'] == 0:
            print(f'validating in epoch {epoch}')
            val_loss, val_metrics = evaluation(model, val_loader, device, default_config, test=False)
            # scheduler.step(val_loss)
            pprint.pprint(val_metrics['endpoint_pred_and_y'][0], sort_dicts=False)
            
            # 保存最佳状态
            if train_helper(val_loss):
                print('update best loss and save model')
                torch.save(model.state_dict(), model_save_path)
            else:
                print('do not update')
            if train_helper.need_stop:
                print('early stopped')
                break
        
    # 测试集评估
    print('evaluating model......')
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_metrics_info = evaluation(model, test_loader, device, default_config = default_config, test=True)
    
    print(f'loss(best model) in test dataset is {test_loss}')
    
    # 输出评价指标，记录预测结果与ground truth
    train_logger.info('prediction and ground truth:')
    for k, v in test_metrics_info.items():
        print(k)
        pprint.pprint(v[0], sort_dicts=False)
        train_logger.info(k)
        train_logger.info(v[0])
        for batch in v[1]:
            for item in batch:
                train_logger.info(item)
            train_logger.info('------------')

    return test_metrics_info

def dataset_to_loader(dataset_name, default_config):
    '''
    加载数据
    '''
    print(f'dataset name is {dataset_name}, loading data...', end=',')
    if default_config['train_settings']['loader'] == 'batch':
        dataset_class = CasInformerDataSetPyG
    else:
        dataset_class = CasInformerDataListPyG
    datasets = dataset_class(dataset_name, default_config['data_settings'], 
                                default_config['model_settings'], 
                                default_config['train_settings'],
                                pre_load = default_config['train_settings']['pre_load'])

    print('splitting data...', end=', ')
    data_length = len(datasets)
    train_size = int(data_length * default_config['train_settings']['train_ratio'])
    val_size = int(data_length * default_config['train_settings']['val_ratio'])
    test_size = data_length - train_size - val_size
    print(f'train_size : {train_size}, val_size:{val_size}, test_size:{test_size}')
    if default_config['train_settings']['random_split']:
        train_dataset, val_dataset, test_dataset = random_split(datasets,
                                                                lengths = [train_size, val_size, test_size],
                                                                generator = torch.Generator().manual_seed(default_config['train_settings']['random_seed']))
    else:
        train_dataset = Subset(datasets, range(train_size))
        val_dataset = Subset(datasets, range(train_size, train_size + val_size))
        test_dataset = Subset(datasets, range(train_size + val_size, data_length))
    
    if default_config['train_settings']['average_batch_data_size']:
        train_dataset = average_batch_data_size(train_dataset, default_config['train_settings']['train_batch_size'])
        val_dataset = average_batch_data_size(val_dataset, default_config['train_settings']['val_batch_size'])
        test_dataset = average_batch_data_size(test_dataset, default_config['train_settings']['test_batch_size'])
    
    if default_config['train_settings']['use_multi_gpu']:
        dataloader = DataListLoaderPyg
    else:
        dataloader = DataLoaderPyg

    train_loader = dataloader(train_dataset, batch_size = default_config['train_settings']['train_batch_size']
                                , shuffle=False, pin_memory = default_config['train_settings']['pin_memory'],
                                drop_last = default_config['train_settings']['drop_last'])
    val_loader = dataloader(val_dataset, batch_size = default_config['train_settings']['val_batch_size']
                                , shuffle=False, pin_memory = default_config['train_settings']['pin_memory'],
                                drop_last = default_config['train_settings']['drop_last'])
    test_loader = dataloader(test_dataset, batch_size = default_config['train_settings']['test_batch_size']
                                , shuffle=False, pin_memory = default_config['train_settings']['pin_memory'],
                                drop_last = default_config['train_settings']['drop_last'])
    
    return train_loader, val_loader, test_loader

def average_batch_data_size(data_sets, batch_size):
    '''
    数据大小不均衡，因此尽量让每个batch的数据大小一致
    '''
    data_sets = list(data_sets)
    data_sets.sort(key=lambda x: x.num_nodes, reverse=True)
    # return data_sets
    total_size = sum([ item.num_nodes for item in data_sets] )
    length = len(data_sets)
    average_size = float(total_size) / length
    # chunk_num = math.ceil(float(length) / batch_size)
    new_list = []
    new_batch_total_size = 0.
    moved = data_sets.pop(0)
    new_list.append(moved)
    new_batch_total_size += moved.num_nodes
    
    while data_sets:
        new_length = len(new_list)
        # if new_length % batch_size == 0:      # 似乎有副作用
        #     new_batch_total_size = 0
        new_average_size = new_batch_total_size / new_length
        if new_average_size <= average_size:
            moved = data_sets.pop(0)
            new_list.append(moved)
            new_batch_total_size += moved.num_nodes
        else:
            moved = data_sets.pop(-1)
            new_list.append(moved)
            new_batch_total_size += moved.num_nodes
    
    # new_list.reverse()
    return new_list

def get_loss(out, y, default_config):
    '''
    
    '''
    loss_type = default_config['train_settings']['loss_type']
    loss_dict = {
        "mae": F.l1_loss,
        "mse": F.mse_loss,
        "msle": lambda x, y: F.mse_loss(torch.log(x+1), torch.log(y+1)),        
        "msle_log2": lambda x, y: F.mse_loss(torch.log2(x+1), torch.log2(y+1)),
        "msle_log10": lambda x, y: F.mse_loss(torch.log10(x+1), torch.log10(y+1)),
        "huber": lambda x, y: F.huber_loss(x, y, delta=300),
        "smooth_l1": lambda x, y: F.smooth_l1_loss(x, y, beta=100),
    }

    if default_config['train_settings']['predict_type'] == 'gap':
        if default_config['model_settings']['rnn_type'] == 'Informer':
            if default_config['train_settings']['use_amp']:
                out = out.float()
                y = y.float()

            # 消融实验时使用，只用末尾时刻的预测值计算loss，以对比不同数据量对模型的性能增益
            # out = out[:,-1]
            # y = y[:,-1]

            return loss_dict[loss_type](F.elu(out) + 1, F.elu(y) + 1)
    
    return loss_dict[loss_type](out, y)

def in_out_pyg_batch(model, batch, device, default_config, eval = False, ):
    '''
    使用pyg的Data对象和dataloader
    '''
    if default_config['train_settings']['use_multi_gpu']:
        batch = [item.to(device) for item in batch]
    else:
        batch = batch.to(device)


    if default_config['train_settings']['use_amp']:
        with torch.cuda.amp.autocast():
            out, y = model(batch)
            batch_loss = get_loss(out, y, default_config)
    else:
        out, y = model(batch)
        batch_loss = get_loss(out, y, default_config)

    train_logger.debug('batch_loss is')
    train_logger.debug(batch_loss)

    if eval:
        return batch_loss, out.detach().cpu().numpy(), y.cpu().numpy()
    return batch_loss

def evaluation(model, data_loader, device, default_config, test = False):
    '''
    评估模型在验证集或训练集上的表现
    '''
    train_logger.debug('before eval, GPU memory ')
    train_logger.debug(torch.cuda.memory_allocated(device))
    train_logger.debug(torch.cuda.memory_reserved(device))
    train_logger.debug(torch.cuda.max_memory_allocated(device))
    model.eval()
    
    sequence_prediction = True if default_config['model_settings']['rnn_type'] == 'Informer' else False
    with torch.no_grad():
        eval_loss = 0.
        all_pred = []       # 预测值
        all_y = []          # 预测值对应的真值
        all_observed_size = []                  # 观察时间观察到的级联大小
        all_y_used_snapshot_in_prediction = []  # 序列预测时使用，每个预测序列中的前几个位置是从观察时间段里取出来的
        all_y_predict_series_pos = []           # 预测任务指定的预测时间在序列中的位置
        all_y_time_interval = []                # 预测序列的每个数值对应时间
        all_cascade_id = []
        
        for batch in tqdm(data_loader):

            batch_loss, pred, y = in_out_pyg_batch(model, batch, device,
                                        default_config = default_config,
                                        eval = True,)
            b_loss = batch_loss.item()
            eval_loss += b_loss if b_loss == b_loss else 0
            
            all_pred.append(pred)
            all_y.append(y)
            if default_config['train_settings']['use_multi_gpu']:
                all_observed_size.extend( [ item.observed_cascade_size.cpu().numpy() for item in batch ])
                all_cascade_id.extend( [item.cascade_id.cpu().numpy() for item in batch])
            else:
                all_observed_size.append(batch.observed_cascade_size.cpu().numpy())
                all_cascade_id.append(batch.cascade_id.cpu().numpy())
            # Informer
            if sequence_prediction:
                # 取出序列预测所需的额外数据
                if default_config['train_settings']['use_multi_gpu']:
                    all_y_used_snapshot_in_prediction.extend([item.y_used_snapshot_in_prediction.cpu().numpy() for item in batch])
                    all_y_predict_series_pos.extend([item.y_predict_series_pos.cpu().numpy() for item in batch])
                    all_y_time_interval.extend( [item.y_time_interval.cpu().numpy() for item in batch] )
                else:
                    all_y_used_snapshot_in_prediction.append(batch.y_used_snapshot_in_prediction.cpu().numpy())
                    all_y_predict_series_pos.append(batch.y_predict_series_pos.cpu().numpy())
                    all_y_time_interval.append(batch.y_time_interval.cpu().numpy())
            
        all_pred = np.concatenate(all_pred)
        # num_of_all_data_in_epoch * (1 or sequence_length)
        all_y = np.concatenate(all_y)
        # num_of_all_data_in_epoch * (1 or sequence_length)
        all_observed_size = np.concatenate(all_observed_size)
        # num_of_all_data_in_epoch * 1
        all_cascade_id = np.concatenate(all_cascade_id)
        # num_of_all_data_in_epoch * 1
        if sequence_prediction:
            all_y_used_snapshot_in_prediction = np.concatenate(all_y_used_snapshot_in_prediction)
            # num_of_all_data_in_epoch * sequence_length
            all_y_predict_series_pos = np.concatenate(all_y_predict_series_pos).tolist()
            # num_of_all_data_in_epoch * 1
            all_y_time_interval = np.concatenate(all_y_time_interval) * default_config['data_settings']['time_coefficient']

    model.train()

    eval_loss = eval_loss / len(data_loader)

    # 所有输出与真值的数值及对应评价指标
    # batch_size = default_config['train_settings']['test_batch_size'] if test else default_config['train_settings']['val_batch_size']
    test_metrics_info = {}
    
    # Informer
    if sequence_prediction:
        # 预测完整规模时，变量名后带gap的表示此时预测出的级联增量
        # 预测级联增量时，变量名后带gap的表示此时预测出的完整规模
        
        all_y_used_snapshot_in_prediction = all_y_used_snapshot_in_prediction.reshape(*all_y.shape)
        all_y_time_interval = all_y_time_interval.reshape(*all_y.shape)
        # num_of_all_data_in_epoch * sequence_length

        unlabeled_prediction = []
        unlabeled_y = []
        unlabeled_prediction_gap = []
        unlabeled_y_gap = []
        unlabeled_cascade_id = []
        endpoint_pred = []
        endpoint_y = []
        endpoint_pred_gap = []
        endpoint_y_gap = []
        endpoint_time = []
        endpoint_cascade_id = []
        unlabeled_prediction_and_y_detail = []
        all_pred_gap = []
        all_y_gap = []
        all_cas_id = []
        
        train_logger.debug('GPU memory after batch, before return')
        train_logger.debug(torch.cuda.memory_allocated(device))
        train_logger.debug(torch.cuda.memory_reserved(device))
        train_logger.debug(torch.cuda.max_memory_allocated(device))

        if default_config['train_settings']['predict_type'] == 'gap':
            # 取出预测值，再加上观察值
            for i in range(len(all_y_used_snapshot_in_prediction)):
                num = all_y_used_snapshot_in_prediction[i].sum()
                bias = all_observed_size[i]
                cas_id = all_cascade_id[i]

                up = all_pred[i][num:]
                unlabeled_prediction.append(up)
                uy = all_y[i][num:]
                unlabeled_y.append(uy)
                upg = all_pred[i][num:] + bias
                unlabeled_prediction_gap.append(upg)
                uyg = all_y[i][num:] + bias
                unlabeled_y_gap.append(uyg)
                ut = all_y_time_interval[i][num:]
                u_id = cas_id.repeat(len(ut))

                pos = int(all_y_predict_series_pos[i])
                endpoint_pred.append(all_pred[i][pos])
                endpoint_y.append(all_y[i][pos])
                endpoint_pred_gap.append(all_pred[i][pos] + bias)
                endpoint_y_gap.append(all_y[i][pos] + bias)
                endpoint_time.append(all_y_time_interval[i][pos])
                endpoint_cascade_id.append(cas_id)

                all_pred_gap.append(all_pred[i] + bias)
                all_y_gap.append(all_y[i] + bias)
                all_cas_id.append(cas_id.repeat(len(all_y[i])))

                unlabeled_prediction_and_y_detail.append(
                    np.concatenate( (u_id.reshape(-1,1),
                                    up.reshape(-1,1), uy.reshape(-1,1),
                                    upg.reshape(-1,1), uyg.reshape(-1,1),
                                    ut.reshape(-1,1)
                                    ) , 
                                    axis=1).reshape(-1,6).tolist()
                )
        else:
            # 取出预测值，再减去观察值
            for i in range(len(all_y_used_snapshot_in_prediction)):
                num = all_y_used_snapshot_in_prediction[i].sum()
                bias = all_observed_size[i]
                cas_id = all_cascade_id[i]

                up = all_pred[i][num:]
                unlabeled_prediction.append(up)
                uy = all_y[i][num:]
                unlabeled_y.append(uy)
                upg = all_pred[i][num:] - bias
                unlabeled_prediction_gap.append(upg)
                uyg = all_y[i][num:] - bias
                unlabeled_y_gap.append(uyg)
                ut = all_y_time_interval[i][num:]
                u_id = cas_id.repeat(len(ut))

                pos = int(all_y_predict_series_pos[i])
                endpoint_pred.append(all_pred[i][pos])
                endpoint_y.append(all_y[i][pos])
                endpoint_pred_gap.append(all_pred[i][pos] - bias)
                endpoint_y_gap.append(all_y[i][pos] - bias)
                endpoint_time.append(all_y_time_interval[i][pos])
                endpoint_cascade_id.append(cas_id)

                all_pred_gap.append(all_pred[i] - bias)
                all_y_gap.append(all_y[i] - bias)
                all_cas_id.append(cas_id.repeat(len(all_y[i])))

                unlabeled_prediction_and_y_detail.append(
                    np.concatenate( (u_id.reshape(-1,1),
                                    upg.reshape(-1,1), uyg.reshape(-1,1),
                                    up.reshape(-1,1), uy.reshape(-1,1),
                                    ut.reshape(-1,1)
                                    ) , 
                                    axis=1).reshape(-1,6).tolist()
                )
        
        unlabeled_prediction = np.concatenate(unlabeled_prediction)
        unlabeled_y = np.concatenate(unlabeled_y)
        unlabeled_prediction_gap = np.concatenate(unlabeled_prediction_gap)
        unlabeled_y_gap = np.concatenate(unlabeled_y_gap)

        endpoint_pred = np.array(endpoint_pred)
        endpoint_y = np.array(endpoint_y)
        endpoint_pred_gap = np.array(endpoint_pred_gap)
        endpoint_y_gap = np.array(endpoint_y_gap)
        endpoint_time = np.array(endpoint_time)
        endpoint_cascade_id = np.array(endpoint_cascade_id)

        all_pred_gap = np.concatenate(all_pred_gap)
        all_y_gap = np.concatenate(all_y_gap)
        all_cas_id = np.concatenate(all_cas_id)

        step = default_config['data_settings']['number_of_snapshot']
        
        if default_config['train_settings']['predict_type'] == 'gap':
            # 预测序列的输出与真值，及对应评价指标
            eval_metrics_unlabeled = prediction_metrics(unlabeled_prediction, unlabeled_y, unlabeled_prediction_gap, unlabeled_y_gap)
            
            # 末尾时刻的输出与真值，及对应评价指标
            endpoint_prediction_and_y_detail = np.concatenate( (endpoint_cascade_id.reshape(-1,1),
                                                            endpoint_pred.reshape(-1,1), endpoint_y.reshape(-1,1),
                                                            endpoint_pred_gap.reshape(-1,1), endpoint_y_gap.reshape(-1,1),
                                                            endpoint_time.reshape(-1,1)
                                                            ) , 
                                                            axis=1).reshape(-1,6).tolist()
            eval_metrics_endpoint = prediction_metrics(endpoint_pred, endpoint_y, endpoint_pred_gap, endpoint_y_gap)

            # 所有时刻的输出与真值，及对应评价指标
            eval_metrics_all = prediction_metrics(all_pred, all_y, all_pred_gap, all_y_gap)
            all_pred_and_y_detail = np.concatenate( (all_cas_id.reshape(-1,1),
                                                    all_pred.reshape(-1,1), all_y.reshape(-1,1),
                                                    all_pred_gap.reshape(-1,1), all_y_gap.reshape(-1,1),
                                                    all_y_time_interval.reshape(-1,1)
                                                    ) 
                                                    , axis=1).reshape(-1,6).tolist()
        
        else:
            # 预测序列的输出与真值，及对应评价指标
            eval_metrics_unlabeled = prediction_metrics(unlabeled_prediction_gap, unlabeled_y_gap, unlabeled_prediction, unlabeled_y)

            # 末尾时刻的输出与真值，及对应评价指标
            endpoint_prediction_and_y_detail = np.concatenate( (endpoint_cascade_id.reshape(-1,1),
                                                            endpoint_pred_gap.reshape(-1,1), endpoint_y_gap.reshape(-1,1),
                                                            endpoint_pred.reshape(-1,1), endpoint_y.reshape(-1,1),
                                                            endpoint_time.reshape(-1,1)
                                                            ) , 
                                                            axis=1).reshape(-1,6).tolist()
            eval_metrics_endpoint = prediction_metrics(endpoint_pred_gap, endpoint_y_gap, endpoint_pred, endpoint_y)

            # 所有时刻的输出与真值，及对应评价指标
            eval_metrics_all = prediction_metrics(all_pred_gap, all_y_gap, all_pred, all_y)
            all_pred_and_y_detail = np.concatenate( (all_cas_id.reshape(-1,1),
                                                    all_pred_gap.reshape(-1,1), all_y_gap.reshape(-1,1),
                                                    all_pred.reshape(-1,1), all_y.reshape(-1,1),
                                                    all_y_time_interval.reshape(-1,1)
                                                    ) 
                                                , axis=1).reshape(-1,6).tolist()

        # 汇总结果
        test_metrics_info["unlabeled_pred_and_y"] = (eval_metrics_unlabeled, unlabeled_prediction_and_y_detail)
        endpoint_prediction_and_y_detail = list([ [item] for item in endpoint_prediction_and_y_detail])
        test_metrics_info["endpoint_pred_and_y"] = (eval_metrics_endpoint, endpoint_prediction_and_y_detail)
        all_pred_and_y_detail = list([ all_pred_and_y_detail[i:i+step] for i in range(0, len(all_pred_and_y_detail), step)])
        test_metrics_info["all_pred_and_y"] = (eval_metrics_all, all_pred_and_y_detail)
    
    # RNN 
    else:
        step = 1
        # predict is size increament
        if default_config['train_settings']['predict_type'] == 'gap':
            
            all_pred_full = []
            all_y_full = []
            for i in range(len(all_observed_size)):
                all_pred_full.append(all_pred[i] + all_observed_size[i])
                all_y_full.append(all_y[i] + all_observed_size[i])
            
            all_pred_full = np.array(all_pred_full)
            all_y_full = np.array(all_y_full)
            all_pred_and_y_detail = np.concatenate( (all_cascade_id.reshape(-1,1),
                                            all_pred.reshape(-1,1), all_y.reshape(-1,1),
                                            all_pred_full.reshape(-1,1), all_y_full.reshape(-1,1)
                                            ), 
                                            axis=1).reshape(-1,5).tolist()
            all_pred_and_y_detail = list([ [item] for item in all_pred_and_y_detail])
            test_metrics_info['endpoint_pred_and_y'] = (prediction_metrics( all_pred, all_y, all_pred_full, all_y_full,), 
                                                    all_pred_and_y_detail)
        # predict is full size
        else:
            all_pred_gap = []
            all_y_gap = []
            for i in range(len(all_observed_size)):
                all_pred_gap.append(all_pred[i] - all_observed_size[i])
                all_y_gap.append(all_y[i] - all_observed_size[i])
            
            all_pred_gap = np.array(all_pred_gap)
            all_y_gap = np.array(all_y_gap)

            all_pred_and_y_detail = np.concatenate( (all_cascade_id.reshape(-1,1),
                                        all_pred_gap.reshape(-1,1), all_y_gap.reshape(-1,1),
                                        all_pred.reshape(-1,1), all_y.reshape(-1,1),
                                        ) , 
                                        axis=1).reshape(-1,5).tolist()
            all_pred_and_y_detail = list([ [item] for item in all_pred_and_y_detail])
            test_metrics_info['endpoint_pred_and_y'] = (prediction_metrics(all_pred_gap, all_y_gap, all_pred, all_y), 
                                                    all_pred_and_y_detail)

    train_logger.debug('GPU memory after eval, ready to return')
    train_logger.debug(torch.cuda.memory_allocated(device))
    train_logger.debug(torch.cuda.memory_reserved(device))
    train_logger.debug(torch.cuda.max_memory_allocated(device))

    return eval_loss, test_metrics_info


if __name__ == '__main__':

    try:
        if not current_platform.startswith('win'):
            multiprocessing.set_start_method('forkserver')
    except RuntimeError:
        # print('RuntimeError raised, cause change multi-process start method in a wrong place')
        pass

    dataset_name = 'acm'
    dataset_name = 'aps'
    dataset_name = 'cascade_sample'
    # dataset_name = 'dblp'
    # dataset_name = 'twitter'
    # dataset_name = 'weibo'
    # dataset_name = 'weibo_lite'

    # 获取参数
    args = get_command_args()

    # 批量训练
    if args.run_all:
        
        results = []
        datasets_to_run = {
            "weibo": [
                0.5,
                1,
                # 2,
                # 3,
            ],
            "twitter": [
                1,
                2,
                # 3,
                # 4,
            ],
            "aps": [
                3,
                5,
                # 7,
                # 9,
            ],
            # "cascade_sample":[0.5, 1, 2, 3]
        }

        current_time = time.time()
        set_logging(filename=f'{"_".join(datasets_to_run.keys())}_{current_time:.0f}_train_{current_platform}.log')
        sys.stdout = Logger(str( log_path / f'{"_".join(datasets_to_run.keys())}_{current_time:.0f}_output_{current_platform}.log'))

        # 遍历所有数据集
        for dataset_name in datasets_to_run.keys():
            try:
                    
                # 设置不同的观察时间
                for ot in datasets_to_run[dataset_name]:
                    
                    # 设置config
                    default_config = get_icp_default_config()
                    dataset_name, default_config = adjust_config_by_args(args, default_config, dataset_name)
                    default_config['data_settings']['observe_time_unit_num'] = ot
                    default_config = adjust_config_auto(default_config)

                    # Informer和RNN两种模式
                    default_config['model_settings']['rnn_type'] = 'Informer'
                    res = train_and_eval(dataset_name, default_config, current_time)
                    res['dataset_name'] = dataset_name
                    res['detail_config'] = default_config
                    results.append( res )

                    default_config['model_settings']['rnn_type'] = 'GRU'
                    res = train_and_eval(dataset_name, default_config, current_time)
                    res['dataset_name'] = dataset_name
                    res['detail_config'] = default_config
                    results.append( res )

            except:
                traceback.print_exc()
            
            res_filename = result_path / f"{'_'.join(datasets_to_run.keys())}_{current_time}_all_results.jsonl"
            with open(res_filename, 'w', encoding='utf8') as f:
                for res in results:
                    f.write( json.dumps(res, indent=2) )

    else:
        # 根据数据集调整设置
        default_config = get_icp_default_config()
        dataset_name, default_config = adjust_config_by_args(args, default_config, dataset_name)
        default_config = adjust_config_auto(default_config)
        # 记录日志
        current_time = time.time()
        set_logging(filename=f'{dataset_name}_{current_time:.0f}_train_{current_platform}.log')
        sys.stdout = Logger(str( log_path / f'{dataset_name}_{current_time:.0f}_output_{current_platform}.log'))
        # 训练
        train_and_eval(dataset_name, default_config, current_time)

    

from pathlib import Path

# 基础目录
base_dir = Path(__file__).resolve().parent.parent
# print(base_dir)

# 设置数据目录
data_path = base_dir / 'data'               # 数据存放根目录
dataset_path = data_path / 'datasets'       # 原始数据集目录
tmp_path = data_path / 'tmp'                # 处理后的数据目录
result_path = data_path / 'result'          # 输出结果目录

# 日志目录
log_path = base_dir / 'log'                

# 设置模型文件目录
model_path = base_dir / 'model'             # 第三方资源及工具包目录

# 根据系统平台设置Java环境变量
import sys
current_platform = sys.platform

import socket
current_host_name = socket.gethostname()


# 级联预测
def get_icp_default_config():
    '''
    信息级联预测算法默认配置
    '''

    hidden_size = 32
    cascade_level_features_dim = 16

    model_settings = dict(

        # gnn_type = 'GCN',
        # gnn_type = 'GAT',
        gnn_type = 'GCN2',
        # gnn_type = 'GAT2',
        gnn_hidden_channels = hidden_size * 2,
        gnn_num_layers = 3,
        gnn_out_channels = hidden_size,
        gnn_dropout = 0.3,

        dynamic_route_iter = 3,
        capsule_out_dim = hidden_size * 2,

        cascade_level_features_dim = cascade_level_features_dim,

        # rnn_type = "LSTM",
        # rnn_type = "GRU",
        rnn_type = "Informer",
        rnn_num_layers = 2,
        rnn_dropout = 0.3,
        rnn_hidden_size = hidden_size * 2,
        # rnn_hidden_size = hidden_size * 3,
        # rnn_hidden_size = hidden_size * 2 + cascade_level_features_dim,
        rnn_bidirectional = True,

        attention_dropout = 0.3,

        mlp_hidden_size = hidden_size * 2,
        mlp_dropout = 0.3,
        mlp_outsize = 1,
        mlp_num_layers = 3,
    )

    data_settings = dict(

        # 节点特征
        dim_of_state_features = 2,      # 状态只有激活与未激活两种
        use_node_state_feature = True,
        dim_of_degree_features = 2,     # 入度和出度
        use_node_degree_feature = True,
        dim_of_time_features = 2,        # 时间维度, 记录初次出现时间和当前时间
        use_node_time_feature = True,
        dim_of_depth_features = 1,      # 节点在此时刻距离根节点的距离
        use_node_depth_feature = False,
        dim_of_node_features = 0,       # 根据上述四个参数计算得来，在此仅用作初始化
        # 级联快照特征
        dim_of_cascade_depth = 1,       # 级联快照时刻，自根节点起至叶节点的最大深度
        use_cascade_depth = True,
        dim_of_cascade_activated_size = 1,  # 级联快照时刻的规模大小
        use_cascade_activated_size = True,
        dim_of_cascade_time = 1,
        use_cascade_time = True,
        dim_of_cascade_features = 0,    # 根据上述两个参数计算得来，在此仅用作初始化
        use_cascade_level_features_in_rnn = False,
        # 级联快照时间特征维度
        time_tuple_features_dim = 7,
        
        number_of_snapshot = 12,                 # 读取级联图快照的数量，过多也不好
        
        sort_by_time = True,
        
        # 设为0时视为不过滤
        observe_time_unit_num = 1,  # 观察多少个单位时间
        observe_time_unit = 3600,   # 每个单位时间对应到数据集中的数字大小
        unit_bias = 0,              # 每隔多少个单位添加bias，如闰年
        predict_time_unit_num = 24, # 预测到第多少个单位时间
        # observe_time_ratio = 0,
        observe_time = 0,           # 由以上参数计算得出，在此仅作初始化，可注释，下同
        predict_time = 0,           # 同上
        time_coefficient = 900,     # 用于数据特征处理，由于数据单位尺度不一，除以系数来放缩大小

        min_size_in_observe_window = 0, # 观察时刻的级联最小规模
        max_size_in_observe_window = 0, # 观察时刻的级联最大规模
        publish_start = 0,              # 级联发布时间不早于
        publish_end = 0,                # 级联发布时间不晚于
    )

    train_settings = dict(

        # learning_rate = 3e-4,
        learning_rate = 0.001,
        weight_decay = 0.01,
        # weight_decay = 0.0001,
        # weight_decay = 1e-5,
        # optimizer_type = "Adam",
        optimizer_type = "AdamW",
        optimizer_eps = 1e-8,
        optimizer_beta1 = 0.9,
        optimizer_beta2 = 0.999,
        optimizer_clip_value = 100.,
        loss_type = "msle",
        # loss_type = 'msle_log2',
        # loss_type = "msle_log10",
        # loss_type = 'mae',
        # loss_type = 'mse',
        # loss_type = 'huber',
        # loss_type = 'smooth_l1',
        
        warmup = True,
        warmup_steps = 0,
        warmup_times = 3,
        init_lr = 1e-7,
        # max_lr = 1e-3,

        reduce_lr = True,
        reduce_lr_patience = 2,
        reduce_lr_factor = 0.1,
        cool_down = 0,
        min_lr = 1e-7,
        reduce_lr_eps = 1e-9,
        
        epochs = 100,
        check_point = 1,
        early_stop = True,
        early_stop_patience = 5,
        early_stop_min_delta = 1e-3,

        device = '0',           # 默认使用GPU:0
        use_multi_gpu = False,  # 默认不多卡训练
        gpu_ids = '0,1',        # 多卡训练默认两张卡
        use_amp = False,        # 混合精度，提高运算速度节省显存，但更容易出现nan
        clean_cache = False,
        
        # predict_type = 'full',     # 预测最终规模
        predict_type = 'gap',      # 预测增量

        loader = 'batch',
        # loader = 'list',
        pre_load = False,
        random_split = False,
        train_ratio = 0.7,
        val_ratio = 0.15,
        test_ratio = 0.15,
        average_batch_data_size = False, 
        
        train_batch_size = 32, 
        val_batch_size = 32,
        test_batch_size = 32,
        drop_last = False,
        pin_memory = False,
        shuffle = False,

        random_seed = 3,
        stable = False,
    )

    dataset_custom_settings = {
        'weibo':dict(
            data = dict(
                observe_time_unit_num = 1,
                predict_time_unit_num = 24,
                time_per_unit = 3600,
                unit_bias = 0,

                # observe_time = 0.5 * 3600,
                observe_time = 1 * 3600,
                # observe_time = 2 * 3600,
                # observe_time = 3 * 3600,
                # observe_time = 9 * 3600,
                predict_time = 24 * 3600,
                time_coefficient = 900,

                min_size_in_observe_window = 10,
                # max_size_in_observe_window = 1000,
                publish_start = 8,
                publish_end = 18,

                # number_of_snapshot = 12,                # 对观察时间为1小时有效
            ),
            model = dict(

            ),
            train = dict(
                train_batch_size = 8, 
                val_batch_size = 8,
                test_batch_size = 8,
                # warmup_steps = 5000,
                average_batch_data_size = False,
            ),
        ),
        'aps':dict(
            data = dict(
                observe_time_unit_num = 3,
                predict_time_unit_num = 20,
                time_per_unit = 365,
                unit_bias = 4,
                
                observe_time = 3 * 365,
                # observe_time = 5 * 365 + 1,
                # observe_time = 7 * 365 + 1,
                # observe_time = 9 * 365 + 1,
                predict_time = 20 * 365 + 5,
                time_coefficient = 30,
                
                publish_start = 1893,
                # publish_end = 1989,
                publish_end = 1997,
                min_size_in_observe_window = 10,

                number_of_snapshot = 10,

                time_tuple_features_dim = 2,
            ),
            model = dict(
                
            ),
            train = dict(
                # init_lr = 3e-5,
                # warmup_steps = 0,
                # warmup_times = 1,

                train_batch_size = 32, 
                val_batch_size = 32,
                test_batch_size = 32,
                # early_stop_patience = 4,
                average_batch_data_size = False,
            ),
        ),
        'twitter':dict(
            data = dict(
                observe_time_unit_num = 1,
                predict_time_unit_num = 32,
                time_per_unit = 3600 * 24,
                unit_bias = 0,

                observe_time = 1 * 3600 * 24,
                # observe_time = 2 * 3600 * 24,
                # observe_time = 3 * 3600 * 24,
                predict_time = 32 * 3600 * 24,
                time_coefficient = 3600,

                publish_end = 410,              # 代表4月10号
                min_size_in_observe_window = 10,

            ),
            model = dict(

            ),
            train = dict(           
                train_batch_size = 16,      # 32显卡放不下，默认16
                val_batch_size = 16,
                test_batch_size = 16,
                # warmup_steps = 5000,
                average_batch_data_size = False,
            ),
        ),
        'cascade_sample':dict(
            data = dict(
                observe_time_unit_num = 1,
                predict_time_unit_num = 24,
                time_per_unit = 3600,
                unit_bias = 0,

                observe_time = 1 * 3600,
                predict_time = 24 * 3600,
                time_coefficient = 60,
                
                min_size_in_observe_window = 10,
                max_size_in_observe_window = 1000,
                publish_start = 8,
                publish_end = 18,
            ),
            model = dict(
                
            ),
            train = dict(
                epochs = 10,
                check_point = 1,
                early_stop_patience = 10,
                train_batch_size = 6,
                val_batch_size = 2,
                test_batch_size = 2,
                average_batch_data_size = False,
            ),
        ),

    }

    default_config = dict(
        model_settings = model_settings,
        data_settings = data_settings,
        train_settings = train_settings,
        dataset_custom_settings = dataset_custom_settings
    )

    return default_config

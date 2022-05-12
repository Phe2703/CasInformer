import sys
import logging
import json
from typing import List
import multiprocessing

from config import log_path, tmp_path, current_platform

# try:
#     if not current_platform.startswith('win'):
#         multiprocessing.set_start_method('forkserver')
# except RuntimeError:
#     # print('RuntimeError raised, cause change multi-process start method in a wrong place')
#     pass

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

def set_logging( filename = 'train_debug', save_path = log_path ):
    save_path.mkdir(exist_ok=True)
    
    train_logger = logging.getLogger('models')
    # 设置logger监控等级
    train_logger.setLevel(logging.DEBUG)
    # 定义一个handler用于写入日志文件
    train_writer = logging.FileHandler(filename= save_path / f'{filename}.log', mode='w', encoding='utf-8')
    # # 定义另一个handler用于向控制台输出
    # train_outputer = logging.StreamHandler()
    # 设置两个handler的监控等级
    train_writer.setLevel(logging.DEBUG)
    # train_outputer.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    # 设置两个handler的日志格式
    train_writer.setFormatter(formatter)
    # train_outputer.setFormatter(formatter)
    # 将两个handler加入到logger中
    train_logger.addHandler(train_writer)
    # train_logger.addHandler(train_outputer)
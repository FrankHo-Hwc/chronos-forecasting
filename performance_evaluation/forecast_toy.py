import torch
import json
import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
sys.path.insert(0, '/public/home/renkan/hwc/exp/chronos-forecasting/src/chronos')
from chronos import ChronosPipeline
from memory_profiler import profile
from data_provider.data_factory import data_provider
from data_provider.metrics import metric
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator,PartialState
from accelerate.utils import gather_object

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return list(obj)
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def test():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='AutoTimes')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, zero_shot_forecasting, in_context_forecasting]')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='chronos_large',
                        help='model name, options: [chronos_large]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, required=True, default='./checkpoints/', help='location of model')
    parser.add_argument('--drop_short', action='store_true', default=False, help='drop too short sequences in dataset')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=576, help='label length')
    parser.add_argument('--token_len', type=int, default=96, help='token length')
    parser.add_argument('--pred_len', type=int, default=96, help='test pred len')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--which_gpu', type=str, default="cuda:0", help='which gpu')
    args = parser.parse_args()

    setting = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.model,
        args.data,
        args.data_path,
        args.root_path,
        args.checkpoints,
        args.seq_len,
        args.pred_len,
        args.batch_size)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

    ZeroShot_Exp(args)
    torch.cuda.empty_cache()

def ZeroShot_Exp(args): 
    accelerator = Accelerator()
    test_data, test_loader = data_provider(args,num_replica = accelerator.num_processes,ranks = accelerator.process_index,flag="test")
    # sampler = DistributedSampler(test_loader.dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
    # test_loader = DataLoader(test_loader.dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    test_loader = tqdm(test_loader, desc=f"Testing,{accelerator.process_index}")
    test_loader = accelerator.prepare(test_loader)
    pipeline = ChronosPipeline.from_pretrained(
        args.checkpoints,
        device_map={"": accelerator.process_index},  # 映射到当前进程对应的 GPU
        torch_dtype=torch.bfloat16,
    )
    
    

    preds_0 = []
    preds_mid = []
    preds_mean = []
    trues = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        print("batch_x:",batch_x.shape)
        forecast = pipeline.predict(
            context=torch.tensor(batch_x.squeeze()),
            prediction_length=args.pred_len,
            num_samples=5,
            limit_prediction_length=False,
        )
        forecast0 = forecast[:, 0, :].squeeze()
        forecast_mid = torch.median(forecast, dim=1).values
        forecast_mean = torch.mean(forecast, dim=1)
        batch_y = batch_y[:, -args.pred_len:].squeeze()
        
        preds_0.append(forecast0)
        preds_mid.append(forecast_mid)
        preds_mean.append(forecast_mean)
        trues.append(batch_y)
                
    preds_0 = torch.cat(preds_0, dim=0)
    preds_mid = torch.cat(preds_mid, dim=0)
    preds_mean = torch.cat(preds_mean, dim=0)
    trues = torch.cat(trues, dim=0)

    
    
    accelerator.wait_for_everyone()
    gathered_preds_0 = gather_object(preds_0)
    gathered_preds_mid = gather_object(preds_mid)
    gathered_preds_mean = gather_object(preds_mean)
    gathered_trues = gather_object(trues)
    
    if accelerator.is_main_process:
        gathered_preds_0 = torch.cat(gathered_preds_0, dim=0).cpu().numpy()
        gathered_preds_mid = torch.cat(gathered_preds_mid, dim=0).cpu().numpy()
        gathered_preds_mean = torch.cat(gathered_preds_mean, dim=0).cpu().numpy()
        gathered_trues = torch.cat(gathered_trues, dim=0).cpu().numpy()

        mae_0, mse_0, _, _, _ = metric(gathered_preds_0, gathered_trues)
        mae_mid, mse_mid, _, _, _ = metric(gathered_preds_mid, gathered_trues)
        mae_mean, mse_mean, _, _, _ = metric(gathered_preds_mean, gathered_trues)
    
        print("mae_0: ", mae_0, "mse_0: ", mse_0)
        print("mae_mid: ", mae_mid, "mse_mid: ", mse_mid)
        print("mae_mean: ", mae_mean, "mse_mean: ", mse_mean)
        
if __name__ == '__main__':
    test()
import torch
import json
import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
from tqdm import tqdm, trange
import data_provider
from chronos import ChronosPipeline
from memory_profiler import profile
from data_provider.data_factory import data_provider
from data_provider.metrics import metric

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
    # parser.add_argument('--test_data_path', type=str, default='ETTh1.csv', help='test data file used in zero shot forecasting')
    parser.add_argument('--checkpoints', type=str, required=True, default='./checkpoints/', help='location of model')
    # parser.add_argument('--drop_last',  action='store_true', default=False, help='drop last batch in data loader')
    # parser.add_argument('--val_set_shuffle', action='store_false', default=True, help='shuffle validation set')
    parser.add_argument('--drop_short', action='store_true', default=False, help='drop too short sequences in dataset')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=576, help='label length')
    parser.add_argument('--token_len', type=int, default=96, help='token length')
    # parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
    # parser.add_argument('--test_label_len', type=int, default=576, help='test label len')
    parser.add_argument('--pred_len', type=int, default=96, help='test pred len')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    # parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # parser.add_argument('--llm_ckp_dir', type=str, default='./llama', help='llm checkpoints dir')
    # parser.add_argument('--mlp_hidden_dim', type=int, default=256, help='mlp hidden dim')
    # parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='mlp hidden layers')
    # parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    # parser.add_argument('--itr', type=int, default=1, help='experiments times')
    # parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    # parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--des', type=str, default='test', help='exp description')
    # parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    # parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    # parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    # parser.add_argument('--weight_decay', type=float, default=0)
    # parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=False)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    args = parser.parse_args()

    # if args.use_multi_gpu:
    #     ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    #     port = os.environ.get("MASTER_PORT", "64209")
    #     hosts = int(os.environ.get("WORLD_SIZE", "8"))
    #     rank = int(os.environ.get("RANK", "0")) 
    #     local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    #     gpus = torch.cuda.device_count()
    #     args.local_rank = local_rank
    #     print(ip, port, hosts, rank, local_rank, gpus)
    #     dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
    #                             rank=rank)
    #     torch.cuda.set_device(local_rank)
    
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
    pipeline = ChronosPipeline.from_pretrained(
        args.checkpoints,
        device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
        torch_dtype=torch.bfloat16,
    )

    preds_0 = []
    preds_mid = []
    preds_mean = []
    trues = []
    test_data, test_loader = data_provider(args, flag="test")
    test_loader = tqdm(test_loader, desc="Testing")
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        forecast = pipeline.predict(
            context=torch.tensor(batch_x.squeeze()),
            prediction_length=args.pred_len,
            num_samples=5,
            limit_prediction_length=False,
        )
        
        forecast = forecast
        forecast0 = forecast[:, 0, :].squeeze()
        forecast_mid = torch.median(forecast, dim=1)[:]#中位数
        forecast_mean = torch.mean(forecast, dim=1)[:]#均值
        batch_y = batch_y[:, -args.pred_len:].squeeze()
        
        preds_0.append(forecast0)
        # print(forecast_mid[0])
        preds_mid.append(forecast_mid[0])
        preds_mean.append(forecast_mean)
        trues.append(batch_y)
        
        # c
        with open('./scripts/test/forecast_log/{}_mid_5.jsonl'.format(args.data), 'a') as outfile:
            for item in forecast_mid[0]:
                json.dump(item.numpy(), outfile, cls = MyEncoder)
                outfile.write('\n')
        with open('./scripts/test/forecast_log/{}_mean_5.jsonl'.format(args.data), 'a') as outfile:
            for item in forecast_mean:
                json.dump(item.numpy(), outfile, cls = MyEncoder)
                outfile.write('\n')
        with open('./scripts/test/forecast_log/{}_original_5.jsonl'.format(args.data), 'a') as outfile:
            for item in forecast:
                json.dump(item.numpy(), outfile, cls = MyEncoder)
                outfile.write('\n')
        with open('./scripts/test/forecast_log/{}_gt_5.jsonl'.format(args.data), 'a') as outfile:
            for item in batch_y:
                json.dump(item.numpy(), outfile, cls = MyEncoder)
                outfile.write('\n')
        # with open('./forecast_log/{}_mse_5.jsonl'.format(args.data), 'a') as outfile:
        #     for item in loss:
        #         json.dump(item, outfile, cls = MyEncoder)
        #         outfile.write('\n')
        if i > 100:
            break       
    preds_0 = torch.cat(preds_0, dim=0).numpy()
    preds_mid = torch.cat(tuple(preds_mid), dim=0).numpy()
    # print(preds_mid[:args.batch_size], preds_mid.shape)
    preds_mean = torch.cat(tuple(preds_mean), dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    mae_0, mse_0, _, _, _ = metric(preds_0, trues)
    mae_mid, mse_mid, _, _, _ = metric(preds_mid, trues)
    mae_mean, mse_mean, _, _, _ = metric(preds_mean, trues)
    
    print("mae_0: ", mae_0, "mse_0: ", mse_0)
    print("mae_mid: ", mae_mid, "mse_mid: ", mse_mid)
    print("mae_mean: ", mae_mean, "mse_mean: ", mse_mean)
    
    # print(forecast.numpy()[0], len(forecast.numpy()))
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # forecast shape: [num_series, num_samples, prediction_length]
        
if __name__ == '__main__':
    test()
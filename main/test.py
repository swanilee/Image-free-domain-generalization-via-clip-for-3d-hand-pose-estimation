from turtle import shape
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--stage', type=str, dest='stage')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--rnn', dest='rnn', action='store_true')
    parser.add_argument('--scale', type=str, default=1.0)
    parser.add_argument('--rot', type=str, default=0.0)
    parser.add_argument('--baseline', type=bool, default=True)
    
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    if not args.stage:
        assert 0, "Please set training stage among [lixel, param]"

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    start = time.time()
    args = parse_args() 
    cfg.set_args(args.gpu_ids, args.stage, False, args.rnn)
    cudnn.benchmark = True

    cfg.scale = float(args.scale)
    cfg.rot = float(args.rot)
    
    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args)
    tester._make_model(args)
    
    eval_result = {}
    outs = {}
    cur_sample_idx = 0
    
    shape_save_arr = list() #np.array([])
    
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        inputs['itr'] = itr
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')

        out = {k: v.cpu().numpy() for k,v in out.items()}
        for k,v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]


        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)

        for k,v in cur_eval_result.items():
            if k in eval_result: eval_result[k] += v
            else: eval_result[k] = v

            
        cur_sample_idx += len(out)

    tester.testset.evaluate_detailed(eval_result,0)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from others.logging import init_logger
from train_abstractive import test_abs_all, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

logger = None


def __str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-checkpoint_path", default='', type=str)
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='/home/hyeonbae/project/kobert_aihub/bert_data/report/report') #../bert_data
    parser.add_argument("-model_path", default='../train_models/MultiSumAbs_report_256')
    parser.add_argument("-result_path", default='../train_results')
    parser.add_argument("-temp_dir", default='../temp')

    parser.add_argument("-batch_size", default=16, type=int) #140
    # parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=256, type=int) #512
    parser.add_argument("-use_interval", type=__str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-large", type=__str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=__str2bool, nargs='?', const=True, default=True) #defralut=False
    parser.add_argument("-lr_bert", default=0.002, type=float) #default=2e-3
    parser.add_argument("-lr_dec", default=0.2, type=float) #default=2e-3
    parser.add_argument("-use_bert_emb", type=__str2bool, nargs='?', const=True, default=True) ##default=False

    parser.add_argument("-share_emb", type=__str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=__str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=210, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=__str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=1, type=int)
    parser.add_argument("-warmup_steps_bert", default=1, type=int)
    parser.add_argument("-warmup_steps_dec", default=1, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=10, type=int)
    parser.add_argument("-accum_count", default=10, type=int)
    parser.add_argument("-report_every", default=10, type=int)
    parser.add_argument("-train_steps", default=20, type=int)
    parser.add_argument("-recall_eval", type=__str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='0', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../train_logs/train_Abs_multi_report_256')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=__str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=__str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=__str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-tokenizer", default='multi', type=str, choices=['multi', 'mecab'])
    parser.add_argument("-vocab", default='', type=str)

    parser.add_argument("-tgt_bos", default='[unused1]', type=str)
    parser.add_argument("-tgt_eos", default='[unused2]', type=str)
    parser.add_argument("-tgt_sent_split", default='[unused3]', type=str)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    args = parser.parse_args()
    device_ids = [int(device_id) for device_id in args.visible_gpus.split(',')]
    args.gpu_ranks = [int(i) for i in range(len(device_ids))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # logger = init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    # logger.info(f'device_ids : {device_ids}, visible_gpus : {args.visible_gpus}, gpu_ranks: {args.gpu_ranks}')
    print(f'device_ids : {device_ids}, visible_gpus : {args.visible_gpus}, gpu_ranks: {args.gpu_ranks}')

    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args, device_ids)

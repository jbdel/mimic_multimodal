import argparse, os, random
import numpy as np
import torch
from MimicDataset import MimicDataset
from visual import *
from doc2vec import Doc2Vec
from torch.utils.data import DataLoader
from train import train

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="Finetune")
    parser.add_argument('--checkpoint', type=str, default=None)

    # Data
    parser.add_argument('--data_root', type=str, default='./data/')
    parser.add_argument('--split_file', type=str, default='mimic-cxr-2.0.0-split.csv')
    parser.add_argument('--label_file', type=str, default='mimic-cxr-2.0.0-chexpert.csv')
    parser.add_argument('--report_folder', type=str, default='mimic-crx-reports/files/')
    parser.add_argument('--image_folder', type=str, default='mimic-crx-images/files/')
    parser.add_argument('--vector_folder', type=str, default='mimic_crx_vectors/')

    # Training
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=99)
    parser.add_argument('--lr_base', type=float, default=0.00005)
    parser.add_argument('--lr_decay', type=float, default=0.2)
    parser.add_argument('--lr_decay_times', type=int, default=2)
    parser.add_argument('--grad_norm_clip', type=float, default=-1)
    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))

    # Dataset and task
    parser.add_argument('--dataset', type=str, default="MimicDataset")


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Base on args given, compute new args
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader
    train_dset = eval(args.dataset)('train', args)
    # eval_dset = eval(args.dataset)('validate', args)
    train_loader = DataLoader(train_dset,
                              1 if args.model == 'Doc2Vec' else args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    # eval_loader = DataLoader(eval_dset, args.batch_size, num_workers=8, pin_memory=True)

    # Net
    net = eval(args.model)(args)

    # Create Checkpoint dir
    os.makedirs(os.path.join(args.output, args.name), exist_ok=True)

    # Run training
    eval_accuracies = train(net, train_loader, None, args)

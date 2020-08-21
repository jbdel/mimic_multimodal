import argparse, os, random
import numpy as np
import torch
from MimicDataset import MimicDataset
from model import Model
from torch.utils.data import DataLoader
from train import train

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="Model")

    # Data


    # Training
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--batch_size', type=int, default=64)
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
    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # eval_loader = DataLoader(eval_dset, args.batch_size, num_workers=8, pin_memory=True)

    # Net
    net = eval(args.model)(args).cuda()
    net.train()
    print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")

    # Create Checkpoint dir
    if not os.path.exists(os.path.join(args.output, args.name)):
        os.makedirs(os.path.join(args.output, args.name))

    # Run training
    eval_accuracies = train(net, train_loader, None, args)

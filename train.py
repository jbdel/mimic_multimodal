import torch
import torch.nn as nn
import time
import numpy as np
import os
import torch.optim as optim

def train(net, train_loader, eval_loader, args):
    loss_fn = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr_base)

    for epoch in range(0, args.max_epoch):
        time_start = time.time()
        loss_sum = 0
        for step, (sample) in enumerate(train_loader):
            loss_tmp = 0
            optimizer.zero_grad()
            pred = net(sample['img'].cuda())
            print(pred.shape)
            sys.exit()
            # loss = loss_fn(pred, ans)
            # loss.backward()
            # optimizer.step()
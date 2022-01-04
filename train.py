import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
import random
import math
from jittor import init
from Rnet import Rnet50
from jittor.dataset import ImageFolder
import jittor.transform as trans
from tensorboardX import SummaryWriter

jt.flags.use_cuda = 1 # if jt.flags.use_cuda = 1 will use gpu

def cust_scheduler(opt, iter, epoch, max_iter, batch_size):
    if(epoch * max_iter + iter) * batch_size >= 50000: 
        opt.lr /= 10
    elif (epoch * max_iter + iter) * batch_size >= 32000: 
        opt.lr /= 10
    

def train(model, train_loader, optimizer, epoch, init_lr, batch_size, writer):
    model.train()
    max_iter = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #cust_scheduler(optimizer, batch_idx, epoch, max_iter, batch_size)
        writer.add_scalar('lr', optimizer.lr, max_iter * epoch + batch_idx)
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        writer.add_scalar('loss', loss.data[0], max_iter * epoch + batch_idx)
        optimizer.step (loss)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0]))


def val(model, val_loader, epoch, writer):
    model.eval()

    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}'.format(epoch, \
                    batch_idx, len(val_loader),100. * float(batch_idx) / len(val_loader), acc))
    print ('Total test acc =', total_acc / total_num)
    writer.add_scalar('test acc', total_acc / total_num, epoch)



def main ():
    batch_size = 128
    init_lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 50
    train_dir = '/mnt/disk/wang/THD-datasets/processed_tsinghuadogs/train'
    val_dir = '/mnt/disk/wang/THD-datasets/processed_tsinghuadogs/val'
    transform = trans.Compose([trans.Resize(224), trans.RandomHorizontalFlip(0.5), trans.ImageNormalize(mean=[0.5], std=[0.5])])
    train_loader = ImageFolder(train_dir, transform=transform).set_attrs(batch_size=batch_size, shuffle=True)
    val_loader = ImageFolder(val_dir, transform=trans.Compose([trans.Resize(224), trans.ImageNormalize(mean=[0.5], std=[0.5])])).set_attrs(batch_size=batch_size, shuffle=True)

    writer = SummaryWriter('runs/exp-2')

    model = Rnet50()
    optimizer = nn.SGD(model.parameters(), init_lr, momentum, weight_decay)
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch, init_lr, batch_size, writer)
        val(model, val_loader, epoch, writer)
    model.save('saved_models/exp-2.pkl')

if __name__ == '__main__':
    main()
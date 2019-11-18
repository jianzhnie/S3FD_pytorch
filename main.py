from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.distributed as dist
import torch.backends.cudnn as cudnn


from data.config import cfg
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
from data.factory import dataset_factory, detection_collate
from models.s3fd import build_s3fd


parser = argparse.ArgumentParser(
    description='S3FD face Detector Training With Pytorch')
parser.add_argument('--dataset',default='face',
                    choices=['hand', 'face', 'head'],
                    help='Train target')
parser.add_argument('--basenet',default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size',default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--pretrained', default=True, type=str,
                    help='use pre-trained model')
parser.add_argument('--resume',default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--distributed', default=True, type=str,
                    help='use distribute training')
parser.add_argument("--local_rank", default=0, type=int)                  
parser.add_argument('--lr', '--learning-rate',default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_folder',default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


cudnn.benchmark = True
args = parser.parse_args()
minmum_loss = np.inf

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def main():
    global args
    global minmum_loss
    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                                init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    # build S3FD 
    print("Building net...")
    s3fd_net = build_s3fd('train', cfg.NUM_CLASSES)
    model = s3fd_net
    if args.pretrained:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Load base network....')
        model.vgg.load_state_dict(vgg_weights)
    
    # for multi gpu
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    model = model.cuda()
    # optimizer and loss function  
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg, args.dataset,True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            minmum_loss = checkpoint['minmum_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print('Initializing weights...')
        model.extras.apply(s3fd_net.weights_init)
        model.loc.apply(s3fd_net.weights_init)
        model.conf.apply(s3fd_net.weights_init)

    print('Loading wider dataset...')
    train_dataset, val_dataset = dataset_factory(args.dataset)

    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True,
                                collate_fn=detection_collate,
                                pin_memory=True)

    val_batchsize = args.batch_size // 2
    val_loader = data.DataLoader(val_dataset, val_batchsize,
                                num_workers=args.num_workers,
                                shuffle=False,
                                collate_fn=detection_collate,
                                pin_memory=True)

    print('Using the specified args:')
    print(args)
    # load PriorBox
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        end = time.time()
        train_loss = train(train_loader, model, priors, criterion, optimizer, epoch)
        val_loss = val(val_loader, model, priors, criterion)
        if args.local_rank == 0:
            is_best = val_loss < minmum_loss
            minmum_loss = min(val_loss, minmum_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': minmum_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch)
        epoch_time = time.time() -end
        print('Epoch %s time cost %f' %(epoch, epoch_time))


def train(train_loader, model, priors, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_loss = AverageMeter()
    cls_loss = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader, 1):
        input, targets = data
        train_loader_len = len(train_loader)

        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = Variable(input.cuda())
        target_var = [Variable(ann.cuda(), requires_grad=False) for ann in targets]

        # compute output
        output = model(input_var)
        loss_l, loss_c = criterion(output, priors, target_var)
        loss = loss_l + loss_c

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            reduced_loss_l = reduce_tensor(loss_l.data)
            reduced_loss_c = reduce_tensor(loss_c.data)
        else:
            reduced_loss = loss.data
            reduced_loss_l = loss_l.data
            reduced_loss_c = loss_c.data
        losses.update(to_python_float(reduced_loss), input.size(0))
        loc_loss.update(to_python_float(reduced_loss_l), input.size(0))
        cls_loss.update(to_python_float(reduced_loss_c), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i >= 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'loc_loss {loc_loss.val:.3f} ({loc_loss.avg:.3f})\t'
                  'cls_loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})'.format(
                   epoch, i, train_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, loc_loss=loc_loss, cls_loss=cls_loss))
    return losses.avg


def val(val_loader, model, priors, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_loss = AverageMeter()
    cls_loss = AverageMeter()

    # switch to train mode
    model.eval()
    end = time.time()

    for i, data in enumerate(val_loader, 1):
        input, targets = data
        val_loader_len = len(val_loader)

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = Variable(input.cuda())
        target_var = [Variable(ann.cuda(), requires_grad=False) for ann in targets]

        # compute output
        output = model(input_var)
        loss_l, loss_c = criterion(output, priors, target_var)
        loss = loss_l + loss_c

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            reduced_loss_l = reduce_tensor(loss_l.data)
            reduced_loss_c = reduce_tensor(loss_c.data)
        else:
            reduced_loss = loss.data
            reduced_loss_l = loss_l.data
            reduced_loss_c = loss_c.data
        losses.update(to_python_float(reduced_loss), input.size(0))
        loc_loss.update(to_python_float(reduced_loss_l), input.size(0))
        cls_loss.update(to_python_float(reduced_loss_c), input.size(0))

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i >= 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'loc_loss {loc_loss.val:.3f} ({loc_loss.avg:.3f})\t'
                  'cls_loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})'.format(
                   i, val_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, loc_loss=loc_loss, cls_loss=cls_loss))
    return losses.avg


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 10

    if epoch >= 30:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 1:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if(args.local_rank == 0 and step % args.print_freq == 0 and step > 1):
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, epoch):
    filename = os.path.join(args.save_folder, "S3FD_" + str(epoch)+ ".pth")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_folder, 'model_best.pth'))

if __name__ == '__main__':
    main()

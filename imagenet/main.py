import argparse
import os
import queue
import time
import threading

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import scatter
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0


class CombinedModel(nn.Module):
    def __init__(self, base):
        super(CombinedModel, self).__init__()
        self.base = base

    def forward(self, x, target):
        x = self.base(x)
        return F.cross_entropy(x, target)


def _thread_main(model, optimizer, in_queue, out_queue):
    def step(input, target):
        loss = model(input, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    while True:
        obj = in_queue.get()
        if obj is None:
            return
        step(obj[0], obj[1])
        out_queue.put(True)


def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    networks = []
    in_queues = []
    out_queues = []
    optimizers = []
    threads = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            # combine network with loss function
            model = CombinedModel(models.__dict__[args.arch]()).cuda()
            networks.append(model)

            # optimizer per GPU
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            optimizers.append(optimizer)

            # queue for input, target pairs
            in_queue = queue.Queue()
            in_queues.append(in_queue)

            # output queue
            out_queue = queue.Queue()
            out_queues.append(out_queue)

            thread = threading.Thread(target=_thread_main, args=(model, optimizer, in_queue, out_queue))
            thread.start()
            threads.append(thread)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(networks, in_queues, out_queues)


def train(networks, in_queues, out_queues):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    for model in networks:
        model.train()

    # create
    input = torch.randn(args.batch_size, 3, 224, 224).pin_memory()
    target = torch.LongTensor(args.batch_size).fill_(1).pin_memory()
    device_ids = list(range(torch.cuda.device_count()))

    def step_all_gpus():
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        scattered_inputs = scatter((input_var, target_var), device_ids)
        for in_queue, input_target in zip(in_queues, scattered_inputs):
            in_queue.put(input_target)

        # Wait until all the kernels are enqueued
        for out_queue in out_queues:
            out_queue.get()

    end = time.time()
    for i in range(5005):
        # measure data loading time
        data_time.update(time.time() - end)

        step_all_gpus()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   1, i, 5005, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        if i == 2:
            # Don't average the first few epochs due to cuDNN benchmarking
            batch_time.reset()


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

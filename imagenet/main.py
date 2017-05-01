import argparse
import os
import shutil
import time
import math
import random
import multiprocessing

import torch
import torch.cuda.nccl2 as nccl2
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import rendezvous


model_names = sorted(
    name for name in models.__dict__
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
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--image-size', default=224, type=int, metavar='N',
                    help='input image size (default: 224)')
parser.add_argument('--central-fraction', default=0.875, type=float,
                    metavar='F',
                    help='crops the central fraction of the image during '
                         'evaluation (default: 0.875)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--accimage', dest='accimage', action='store_true',
                    help='use accimage library')
parser.add_argument('-n', '--num-replicas', metavar='N', type=int,
                    help='number of replicas')
parser.add_argument('--ttl', metavar='TTL', type=int, default=1,
                    help='TTL for multicast discovery')
parser.add_argument('--checkpoint-dir', metavar='DIR', default='.',
                    help='checkpoint directory')

best_prec1 = 0
rank = None
copy_stream = None


def main():
    global args, rank
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    rank, device = rendezvous.rendezvous(args.num_replicas, args.ttl)
    with torch.cuda.device(device):
        main2()


def main2():
    global best_prec1

    # accimage
    if args.accimage:
        torchvision.set_image_backend('accimage')
    else:
        print('WARNING: using PIL image loader (you may want --accimage)')

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model.cuda()
    for param in model.parameters():
        nccl2.broadcast(param.data, root=0)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        SubsampleDataset(datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])), rank, args.num_replicas),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(round(args.image_size / args.central_fraction)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            # average out batch norm mean and var across replicas
            average_batch_norm_stats(model)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    all_reduce_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = copy_inputs(input, target)

        # compute output
        loss = criterion(model(input), target)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        all_reduce_grads(model, all_reduce_time)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'All Reduce {all_reduce.val:.3f} ({all_reduce.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, all_reduce=all_reduce_time, loss=losses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = copy_inputs(input, target, volatile=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def copy_inputs(*tensors, **kwargs):
    """Copies tensors on a background stream to the GPU"""
    global copy_stream
    if copy_stream is None:
        copy_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    outputs = []
    with torch.cuda.stream(copy_stream):
        for tensor in tensors:
            outputs.append(torch.autograd.Variable(tensor.cuda(async=True), **kwargs))

    default_stream.wait_stream(copy_stream)
    for output in outputs:
        output.data.record_stream(default_stream)

    return tuple(outputs)


def average_batch_norm_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nccl2.all_reduce(m.running_mean, output=m.running_mean)
            nccl2.all_reduce(m.running_var, output=m.running_var)
            m.running_mean /= args.num_replicas
            m.running_var /= args.num_replicas


events = []


def all_reduce_grads(model, all_reduce_time):
    if len(events) == 0:
        events.extend([
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        ])
    elif events[1].query():
        elapsed_time = events[0].elapsed_time(events[1])
        all_reduce_time.update(elapsed_time / 1000.0)

    events[0].record()
    buffer_size = 10485760
    scale_factor = args.batch_size / 256.0
    params = list(model.parameters())
    for chunk in torch.cuda.comm._take_tensors(params, buffer_size):
        tensors = [p.grad.data for p in chunk]
        if len(tensors) == 1:
            nccl2.all_reduce(tensors[0])
            tensors[0] *= scale_factor
        else:
            buf = torch.cat([t.contiguous().view(-1) for t in tensors], 0)
            buf *= scale_factor
            nccl2.all_reduce(buf)
            offset = 0
            for t in tensors:
                numel = t.numel()
                t.set_(buf[offset:offset+numel].view_as(t))
                offset += numel
    events[1].record()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(args.checkpoint_dir, filename)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(args.checkpoint_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SubsampleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rank, n):
        self.dataset = dataset
        self.size = math.ceil(len(dataset) / n)

    def __getitem__(self, i):
        i = random.randint(0, len(self.dataset) - 1)
        return self.dataset[i]

    def __len__(self):
        return self.size


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()

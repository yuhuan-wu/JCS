import os, shutil
import torch
import pickle
from Models import single_model as net
import numpy as np
import Transforms as myTransforms
from Dataset import Dataset
from parallel import DataParallelModel, DataParallelCriterion
import time
from argparse import ArgumentParser
from IoUEval import IoUEval
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
from torch.nn.parallel import gather
import torch.nn as nn
import torch.nn.functional as F



def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs, target):
        if isinstance(target, tuple):
            target = target[0]
        if inputs.shape[1] == 5:
            loss1 = BCEDiceLoss(inputs[:, 0, :, :], target)
            loss2 = BCEDiceLoss(inputs[:, 1, :, :], target)
            loss3 = BCEDiceLoss(inputs[:, 2, :, :], target)
            loss4 = BCEDiceLoss(inputs[:, 3, :, :], target)
            loss5 = BCEDiceLoss(inputs[:, 4, :, :], target)
            return loss1 + loss2 + loss3 + loss4 + loss5
        elif inputs.shape[1] == 1:
            #print(inputs.shape)
            loss = BCEDiceLoss(inputs[:, 0, :, :], target)
            return loss


class FLoss(nn.Module):
    def __init__(self, beta=0.3, log_like=False):
        super(FLoss, self).__init__()
        self.beta = beta
        self.log_like = log_like

    def _compute_loss(self, prediction, target):
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            loss = -torch.log(fmeasure)
        else:
            loss  = 1 - fmeasure
        return loss.mean()

    def forward(self, inputs, target):
        loss1 = self._compute_loss(inputs[:, 0, :, :], target)
        loss2 = self._compute_loss(inputs[:, 1, :, :], target)
        loss3 = self._compute_loss(inputs[:, 2, :, :], target)
        loss4 = self._compute_loss(inputs[:, 3, :, :], target)
        loss5 = self._compute_loss(inputs[:, 4, :, :], target)
        return 1.0*loss1 + 1.0*loss2 + 1.0*loss3 + 1.0*loss4 + 1.0*loss5



@torch.no_grad()
def val(args, val_loader, model, criterion):
    # switch to evaluation mode
    model.eval()
    sal_eval_val = IoUEval()
    epoch_loss = []
    total_batches = len(val_loader)
    for iter, (input, target) in enumerate(val_loader):
        start_time = time.time()

        if args.gpu:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        output = model(input_var)
        loss = criterion(output, target_var)
        #torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.gpu and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)
        sal_eval_val.add_batch(output[:, 0, :, :],  target_var)
        if iter % 50 == 0 or iter == len(val_loader) - 1:
            print('[%d/%d] loss: %.3f time: %.3f' % (iter, total_batches, loss.data.item(), time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    IoU, MAE = sal_eval_val.get_metric()

    return average_epoch_loss_val, IoU, MAE


def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0):
    # switch to train mode
    model.eval()
    sal_eval_train = IoUEval()
    epoch_loss = []
    total_batches = len(train_loader)
    for iter, (input, target) in enumerate(train_loader):
        start_time = time.time()

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches)

        if args.gpu == True:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()

        # run the model
        output = model(input_var)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time

        # compute the confusion matrix
        if args.gpu and torch.cuda.device_count() > 1:
           output = gather(output, 0, dim=0)
        with torch.no_grad():
            sal_eval_train.add_batch(output[:, 0, :, :] , target_var)
        if iter % 20 == 0 or iter == len(train_loader) - 1:
            print('[%d/%d] iteration: [%d/%d] lr: %.7f loss: %.3f time:%.3f' % (iter, \
                    total_batches, iter+cur_iter, max_batches*args.max_epochs, lr, \
                    loss.data.item(), time_taken))
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    IoU, MAE = sal_eval_train.get_metric()

    return average_epoch_loss_train, IoU, MAE, lr


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200: # warm up
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_validate_covid(args):
    # load the model
    model = net.JCS(pretrained='model_zoo/5stages_vgg16_bn-6c64b313.pth')

    args.savedir = args.savedir + '/'
    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if True:
        print('copying train.py, train.sh, EDN_train.py to snapshots dir')
        shutil.copy('tools/train_single.py', args.savedir + 'train.py')
        shutil.copy('tools/train.sh', args.savedir + 'train.sh')
        shutil.copy('Models/single_model.py', args.savedir + 'single_model.py')
        shutil.copy('Models/utils.py', args.savedir + 'utils.py')
    if args.gpu and torch.cuda.device_count() > 1:
        #model = nn.DataParallel(model)
        model = DataParallelModel(model)

    if args.gpu:
        model = model.cuda()

    total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_paramters))

    NORMALISE_PARAMS = [np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape((1, 1, 3)), # MEAN
                        np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape((1, 1, 3))] # STD

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(args.width, args.height),
        myTransforms.RandomCropResize(int(7./224.*args.width)),
        myTransforms.RandomFlip(),
        #myTransforms.GaussianNoise(),
        myTransforms.ToTensor()
    ])

    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        #myTransforms.Scale(512, 512),
        myTransforms.Scale(352, 352),
        myTransforms.RandomCropResize(int(7./224.*args.width)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])
    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        #myTransforms.Scale(1024, 1024),
        myTransforms.Scale(448, 448),
        myTransforms.RandomCropResize(int(7./224.*args.width)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(args.width, args.height),
        myTransforms.ToTensor()
    ])

    # since we training from scratch, we create data loaders at different scales
    # so that we can generate more augmented data and prevent the network from overfitting
    trainLoader_main = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'train', transform=trainDataset_main),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    valLoader = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'test', transform=valDataset),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    max_batches = len(trainLoader_main) #+ len(trainLoader_scale1) + len(trainLoader_scale2)
    print('max_batches {}'.format(max_batches))
    if args.gpu:
        cudnn.benchmark = True

    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            #args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    log_file = args.savedir + args.log_file
    if os.path.isfile(log_file):
        logger = open(log_file, 'a')
    else:
        logger = open(log_file, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t\t%s\t%s\t%s\t%s\t%s\tlr" % ('Epoch', \
                'Loss(Tr)', 'IoU (tr)', 'MAE (tr)', 'IoU (val)', 'MAE (val)'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    cur_iter = 0

    criteria = CrossEntropyLoss()
    if args.gpu and torch.cuda.device_count() > 1:
        criteria = DataParallelCriterion(criteria)

    for epoch in range(start_epoch, args.max_epochs):
        # train for one epoch
        loss_tr, IoU_tr, MAE_tr, lr = \
            train(args, trainLoader_main, model, criteria, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader_main)
        torch.cuda.empty_cache()

        # evaluate on validation set
        loss_val, IoU_val, MAE_val = val(args, valLoader, model, criteria)
        torch.cuda.empty_cache()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_tr': loss_tr,
            'loss_val': loss_val,
            'iou_tr': IoU_tr,
            'iou_val': IoU_val,
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        # save the model also
        model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
        if IoU_val > 0.7:
            print("found a good model > 0.7, start to save that!")
            torch.save(model.state_dict(), model_file_name)

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, loss_tr, IoU_tr, MAE_tr, IoU_val, MAE_val, lr))
        logger.flush()
        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t IoU(tr) = %.4f\t IoU(val) = %.4f" \
                % (epoch, loss_tr, loss_val, IoU_tr, IoU_val))
        torch.cuda.empty_cache()
    logger.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=10, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
    parser.add_argument('--log_file', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    train_validate_covid(args)

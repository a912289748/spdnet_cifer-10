import argparse
from time import time

import torchvision
import torchvision.transforms as transforms


import numpy as np
import random
import h5py
import os

import multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

import model
from torch.autograd import Variable
import torch
import datetime
import torch.nn.functional as F
import matplotlib

from utils.GetBci2a import getAllDataloader

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

if __name__ == '__main__':
    logger = SummaryWriter(log_dir="data/log")

    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
    ap.add_argument('--sub', type=int, default=1, help='subjectxx you want to triain')
    ap.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-1, help='weight decay')
    ap.add_argument('--iterations', type=int, default=350, help='number of training iterations')
    ap.add_argument('--epochs', type=int, default=3, help='number of epochs that you want to use for split EEG signals')
    ap.add_argument('--bs', type=int, default=128, help='batch size')
    ap.add_argument('--model_path', type=str, default='./checkpoint/bci2a/',
                    help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default='data/BCICIV_2a_mat/', help='data path')
    args = vars(ap.parse_args())

    print(f'subject{args["sub"]}')
    trainloader, validloader, testloader = getAllDataloader(subject=args['sub'],
                                                            ratio=8,
                                                            data_path=args['data_path'],
                                                            bs=args['bs'])


    batch_size = 8192#别忘修改 是个坑
    lr = 0.1
    num_epoch = 300000



    #手动载入模型
    # model = torch.load("C:\\Users\\l\\Downloads\\spdnetxiugai\\SPDNet-master\\tmp\\cifar-10\\cifar_9c.model")
    model = model.SPDNetwork()
    hist_loss = []
    for epoch in range(num_epoch):
        # shuffled_index = list(range(len(train_file)))
        random.seed(6666)
        # random.shuffle(shuffled_index)

        # train_file = [train_file[i] for i in shuffled_index]
        # train_label = [train_label[i] for i in shuffled_index]
        #
        # for idx_train in range(0, len(train_file) // batch_size):
        #     idx = idx_train
        #     b_file = train_file[idx * batch_size:(idx + 1) * batch_size]
        #     b_label = train_label[idx * batch_size:(idx + 1) * batch_size]
        #     batch_data = np.zeros([batch_size, 400, 400], dtype=np.float32)
        #     batch_label = np.zeros([batch_size], dtype=np.int32)
        #     i = 0
        #     for file in b_file:
        #         # f = h5py.File(os.path.join(data_dir, file), 'r')
        #         spd = sio.loadmat(os.path.join(data_dir, file))['Y1']
        #         batch_data[i, :, :] = spd
        #         batch_label[i] = int(b_label[i]) - 1
        #         i += 1
        #
        #     input = Variable(torch.from_numpy(batch_data)).double()
        #     target = Variable(torch.LongTensor(batch_label))
        all_correct_train = 0
        all_loss_train = 0
        for i, (images, labels) in enumerate(trainloader):
            # images = images.squeeze(1)
            # labels 的标签是[batch,0,类别个数，第几个类]需要转换成 [batch,]
            input = images
            target = labels
            target = torch.nn.functional.one_hot(labels, num_classes=10)

            stime = datetime.datetime.now()
            logits = model(input)#30 400 400
            logits = logits.squeeze(0)
            output = F.log_softmax(logits,dim=-1)
            loss = F.nll_loss(output, labels)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(labels.data.view_as(pred)).long().cpu().sum()
            all_correct_train  = all_correct_train + correct
            loss.backward()
            model.update_para(lr)
            etime = datetime.datetime.now()
            dtime = etime.second - stime.second

            hist_loss.append(loss.data)
            all_loss_train = all_loss_train + loss

            print('[epoch %d/%d] [iter %d/%d] loss %f acc %f %f/batch' % (epoch, num_epoch,
                                                            i, len(trainloader), loss.data,
                                                             correct/batch_size, dtime))
        print("step: %d [all accurate %f] " %(epoch,all_correct_train.data/(len(trainloader)*batch_size)))



        # 验证
        all_correct_val = 0
        all_loss_val = 0
        for i, (images, labels) in enumerate(testloader):
            input = images
            target = labels
            target = torch.nn.functional.one_hot(labels, num_classes=10)

            stime = datetime.datetime.now()
            logits = model(input)  # 30 400 400
            logits = logits.squeeze(0)
            output = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(output, labels)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(labels.data.view_as(pred)).long().cpu().sum()
            all_correct_val = all_correct_val + correct
            # loss.backward()
            # model.update_para(lr)
            etime = datetime.datetime.now()
            dtime = etime.second - stime.second

            hist_loss.append(loss.data)
            all_loss_val = all_loss_val + loss

            print('[epoch %d/%d] [iter %d/%d] loss %f acc %f %f/batch' % (epoch, num_epoch,
                                                                          i, len(testloader), loss.data,
                                                                          correct / batch_size, dtime))
        print("step: %d [all accurate %f] " % (epoch, all_correct_val.data / (len(trainloader) * batch_size)))
        logger.add_scalars("loss_1", tag_scalar_dict={"train_loss":all_loss_train/(len(trainloader)*batch_size),"val_loss":all_loss_val/(len(testloader)*batch_size)},global_step=epoch)
        logger.add_scalars("acc_1",tag_scalar_dict={"train_acc":all_correct_train/(len(trainloader)*batch_size),"val_acc":all_correct_val/(len(testloader)*batch_size)} ,global_step=epoch)
        torch.save(model, './tmp/cifar-10/cifar_' + str(epoch + 105) + 'c.model')

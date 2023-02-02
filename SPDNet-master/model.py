import torch
from torch.autograd import Variable
import scipy.io as sio
import numpy as np
import spd_net_util as util
import functorch
from torch import nn
from scipy.linalg import orth

class SPDNetwork(torch.nn.Module):

    def __init__(self):
        super(SPDNetwork, self).__init__()
        self.covar =Covariance()

        tmp = sio.loadmat('./tmp/afew/w_1.mat')['w_1']
        self.w_1_p = Variable(torch.from_numpy(tmp), requires_grad=True)
        u, v, s = torch.svd(torch.FloatTensor(np.random.randn(32, 16).astype(np.float32)))
        u = u.unsqueeze(0)
        #32->16  16->8  8->4
        self.weight_1 = nn.Parameter(u, requires_grad=True)

        u, v, s = torch.svd(torch.FloatTensor(np.random.randn(16, 8).astype(np.float32)))
        u = u.unsqueeze(0)
        self.weight_2 = nn.Parameter(u, requires_grad=True)

        u, v, s = torch.svd(torch.FloatTensor(np.random.randn(8, 4).astype(np.float32)))
        u = u.unsqueeze(0)
        self.weight_3 = nn.Parameter(u, requires_grad=True)
        # self.weight_1 = self.weight_1.unsqueeze(0)
        # self.weight_2 = self.weight_2.unsqueeze(0)
        # self.weight_3 = self.weight_3.unsqueeze(0)
        self.weight_fc = nn.Parameter(torch.FloatTensor(np.random.randn(16,10)).unsqueeze(0),requires_grad=True)
        tmp = sio.loadmat('./tmp/afew/w_2.mat')['w_2']
        self.w_2_p = Variable(torch.from_numpy(tmp), requires_grad=True)

        tmp = sio.loadmat('./tmp/afew/w_3.mat')['w_3']
        self.w_3_p = Variable(torch.from_numpy(tmp), requires_grad=True)

        tmp = sio.loadmat('./tmp/afew/fc.mat')['theta']
        self.fc_w = Variable(torch.from_numpy(tmp.astype(np.float64)), requires_grad=True)

    def forward(self, input):
        batch_size = input.shape[0]
        w_1_pc = self.w_1_p.contiguous()
        w_1 = w_1_pc.view([1, w_1_pc.shape[0], w_1_pc.shape[1]])

        w_2_pc = self.w_2_p.contiguous()
        w_2 = w_2_pc.view([1,w_2_pc.shape[0], w_2_pc.shape[1]])

        w_3_pc = self.w_3_p.contiguous()
        w_3 = w_3_pc.view([1, w_3_pc.shape[0], w_3_pc.shape[1]])
        spd_matrix = self.covar(input)

        res_1 = torch.matmul(torch.transpose(self.weight_1, dim0=1, dim1=2), spd_matrix)
        res_1 = torch.matmul(res_1, self.weight_1)
        res_1 = util.rec_mat_v2(res_1)

        res_2 = torch.matmul(torch.transpose(self.weight_2, dim0=1, dim1=2), res_1)
        res_2 = torch.matmul(res_2,self.weight_2)
        res_2 = util.rec_mat_v2(res_2)

        res_3 = torch.matmul(torch.transpose(self.weight_3, dim0=1, dim1=2), res_2)
        res_3 = torch.matmul(res_3, self.weight_3)
        res_3 = util.rec_mat_v2(res_3)

        # w_tX = torch.matmul(torch.transpose(self.weight_1, dim0=1, dim1=2), input)
        # w_tXw = torch.matmul(self.weight_1, w_1)
        # X_1 = util.rec_mat_v2(w_tXw)
        logits = torch.matmul(res_3.view([batch_size, -1]), self.weight_fc)

        # w_tX = torch.matmul(torch.transpose(w_2, dim0=1, dim1=2), X_1)
        # w_tXw = torch.matmul(w_tX, w_2)
        # X_2 = util.rec_mat_v2(w_tXw)
        #
        # w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), X_2)
        # w_tXw = torch.matmul(w_tX, w_3)
        # X_3 = util.log_mat_v2(w_tXw)
        #
        # feat = X_3.view([batch_size, -1])  # [batch_size, d]
        # logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class]

        return logits

    def update_para(self, lr):

        # egrad_w1 = self.w_1_p.grad.data.numpy()
        egrad_w1 = self.weight_1.grad.data.squeeze(0).numpy()
        egrad_w2 = self.weight_2.grad.data.squeeze(0).numpy()
        egrad_w3 = self.weight_3.grad.data.squeeze(0).numpy()
        # egrad_w2 = self.w_2_p.grad.data.numpy()
        # egrad_w3 = self.w_3_p.grad.data.numpy()
        w_1_np = self.weight_1.data.squeeze(0).numpy()
        w_2_np = self.weight_2.data.squeeze(0).numpy()
        w_3_np = self.weight_3.data.squeeze(0).numpy()
        # w_1_np = self.w_1_p.data.numpy()
        # w_2_np = self.w_2_p.data.numpy()
        # w_3_np = self.w_3_p.data.numpy()

        new_w_3 = util.update_para_riemann(w_3_np, egrad_w3, lr)
        new_w_2 = util.update_para_riemann(w_2_np, egrad_w2, lr)
        new_w_1 = util.update_para_riemann(w_1_np, egrad_w1, lr)

        # print(np.sum(w_1_np))
        # print(np.sum(np.square(w_3_np - new_w_3)))
        # print(np.sum(np.square(w_2_np - new_w_2)))
        # print(np.sum(np.square(w_1_np - new_w_1)))

        self.weight_1.data.copy_(torch.DoubleTensor(new_w_1).unsqueeze(0))
        self.weight_2.data.copy_(torch.DoubleTensor(new_w_2).unsqueeze(0))
        self.weight_3.data.copy_(torch.DoubleTensor(new_w_3).unsqueeze(0))
        # self.w_2_p.data.copy_(torch.DoubleTensor(new_w_2))
        # self.w_3_p.data.copy_(torch.DoubleTensor(new_w_3))

        self.weight_fc.data -= lr * self.weight_fc.grad.data
        # Manually zero the gradients after updating weights
        self.weight_1.grad.data.zero_()
        self.weight_2.grad.data.zero_()
        self.weight_3.grad.data.zero_()
        self.weight_fc.grad.data.zero_()
        # print('finished')


class Covariance(torch.nn.Module):

    def __init__(self, append_mean=True):
        super(Covariance, self).__init__()
        self.append_mean = append_mean

    def forward(self, input):
        batch_size = input.shape[0]

        mean = torch.mean(input, 2, keepdim=True)
        x = input - mean.expand(-1, -1, 32,-1)
        x_1 = x.squeeze(1)
        x_2 = x.transpose(2,3).squeeze(1)
        output = torch.bmm(x_1,x_2) / input.size(2)
        #循环batch_size次，可以优化，
        for i in range(batch_size):#
            output[i]=output[i] + (functorch.vmap(torch.trace)(output)[i] * 0.01)


        # # torch.cat(torch.mul(torch.eye(32), (functorch.vmap(torch.trace)(output)[0] * 0.01)),torch.mul(torch.eye(32), (functorch.vmap(torch.trace)(output)[1] * 0.01)),torch.mul(torch.eye(32), (functorch.vmap(torch.trace)(output)[2] * 0.01)),torch.mul(torch.eye(32), (functorch.vmap(torch.trace)(output)[3] * 0.01)))
        # if self.append_mean:
        #         mean_sq = torch.bmm(mean, mean.transpose(1, 2))
        #         output.add_(mean_sq)
        #         output = torch.cat((output, mean), 2)
        #         one = input.new(1, 1, 1).fill_(1).expand(mean.size(0), -1, -1)
        #         mean = torch.cat((mean, one), 1).transpose(1, 2)
        #         output = torch.cat((output, mean), 1)

        return output
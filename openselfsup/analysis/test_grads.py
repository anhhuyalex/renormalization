import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pdb


def grad_check_wo_l2_norm():
    i = Variable(torch.randn(1, 5), requires_grad=True)

    w = Variable(torch.randn(5, 10), requires_grad=True)
    b = Variable(torch.randn(1, 10), requires_grad=True)

    w2 = Variable(torch.randn(10, 15), requires_grad=True)
    b2 = Variable(torch.randn(1, 15), requires_grad=True)

    t = Variable(torch.randn(1, 15), requires_grad=True)
    tanh = nn.Tanh()
    tanh_out = tanh(torch.matmul(i, w) + b)
    tanh_out2 = tanh(torch.matmul(tanh_out, w2) + b2)
    loss = -torch.sum(tanh_out2 * t)

    self_grad_b2 = -t * (1 - tanh_out2**2)
    self_grad_w2 = torch.matmul(tanh_out.transpose(1, 0), self_grad_b2)
    self_grad_b = (1 - tanh_out**2) * torch.matmul(self_grad_b2, w2.transpose(1, 0))
    self_grad_w = torch.matmul(i.transpose(1, 0), self_grad_b)
    loss.backward()

    allclose_kwargs = dict(
            rtol=1.e-4, atol=1.e-5
            )
    print('W2 grad coorect? ', np.allclose(self_grad_w2.detach().numpy(), w2.grad.numpy(), **allclose_kwargs))
    print('B2 grad coorect? ', np.allclose(self_grad_b2.detach().numpy(), b2.grad.numpy(), **allclose_kwargs))
    print('W grad coorect? ', np.allclose(self_grad_w.detach().numpy(), w.grad.numpy(), **allclose_kwargs))
    print('B grad coorect? ', np.allclose(self_grad_b.detach().numpy(), b.grad.numpy(), **allclose_kwargs))


def grad_check_with_l2_norm():
    i = Variable(torch.randn(1, 5), requires_grad=True)

    w = Variable(torch.randn(5, 10), requires_grad=True)
    b = Variable(torch.randn(1, 10), requires_grad=True)

    w2 = Variable(torch.randn(10, 15), requires_grad=True)
    b2 = Variable(torch.randn(1, 15), requires_grad=True)

    t = Variable(torch.randn(1, 15), requires_grad=True)
    tanh = nn.Tanh()
    tanh_out = tanh(torch.matmul(i, w) + b)
    out2 = torch.matmul(tanh_out, w2) + b2
    l2_norm = torch.sum(out2 ** 2)
    l2_normed_out2 = out2 / l2_norm
    loss = -torch.sum(l2_normed_out2 * t)

    self_grad_b2 = \
            -t / l2_norm \
            + torch.sum(t * out2) / (l2_norm**2) \
              * 2 * out2
    self_grad_w2 = torch.matmul(tanh_out.transpose(1, 0), self_grad_b2)
    self_grad_b = (1 - tanh_out**2) * torch.matmul(self_grad_b2, w2.transpose(1, 0))
    self_grad_w = torch.matmul(i.transpose(1, 0), self_grad_b)
    loss.backward()

    allclose_kwargs = dict(
            rtol=1.e-4, atol=1.e-5
            )
    print('B2 grad coorect? ', np.allclose(self_grad_b2.detach().numpy(), b2.grad.numpy(), **allclose_kwargs))
    print('W2 grad coorect? ', np.allclose(self_grad_w2.detach().numpy(), w2.grad.numpy(), **allclose_kwargs))
    print('B grad coorect? ', np.allclose(self_grad_b.detach().numpy(), b.grad.numpy(), **allclose_kwargs))
    print('W grad coorect? ', np.allclose(self_grad_w.detach().numpy(), w.grad.numpy(), **allclose_kwargs))


def grad_check_with_l2_norm_with_bs():
    i = Variable(torch.randn(4, 5), requires_grad=True)
    w = Variable(torch.randn(4, 5, 10), requires_grad=True)
    b = Variable(torch.randn(4, 10), requires_grad=True)
    w2 = Variable(torch.randn(4, 10, 15), requires_grad=True)
    b2 = Variable(torch.randn(4, 15), requires_grad=True)
    t = Variable(torch.randn(4, 15), requires_grad=True)
    tanh = nn.Tanh()

    tanh_out = tanh(
            torch.einsum(
                'bpq, bqo->bpo',
                i.unsqueeze(1), w).squeeze(1) \
            + b)
    out2 = \
            torch.einsum(
                'bpq, bqo->bpo',
                tanh_out.unsqueeze(1), w2).squeeze(1) \
           + b2
    l2_norm = torch.sum(out2 ** 2, dim=-1, keepdim=True)
    l2_normed_out2 = out2 / l2_norm
    loss = -torch.sum(l2_normed_out2 * t, dim=-1)

    self_grad_b2 = \
            -t / l2_norm \
            + torch.sum(t * out2, dim=-1, keepdim=True) / (l2_norm**2) \
              * 2 * out2
    self_grad_w2 = torch.einsum(
            'bpq, bqo->bpo',
            tanh_out.unsqueeze(-1), self_grad_b2.unsqueeze(1))
    self_grad_b = \
            (1 - tanh_out**2) \
            * torch.einsum(
                'bpq, bqo->bpo',
                self_grad_b2.unsqueeze(1), w2.transpose(2, 1)
                ).squeeze(1)
    self_grad_w = torch.einsum(
            'bpq, bqo->bpo',
            i.unsqueeze(-1), self_grad_b.unsqueeze(1))
    _i = 2
    loss[_i].backward()

    allclose_kwargs = dict(
            rtol=1.e-4, atol=1.e-5
            )
    print('B2 grad coorect? ', np.allclose(self_grad_b2[_i].detach().numpy(), b2.grad[_i].numpy(), **allclose_kwargs))
    print('W2 grad coorect? ', np.allclose(self_grad_w2[_i].detach().numpy(), w2.grad[_i].numpy(), **allclose_kwargs))
    print('B grad coorect? ', np.allclose(self_grad_b[_i].detach().numpy(), b.grad[_i].numpy(), **allclose_kwargs))
    print('W grad coorect? ', np.allclose(self_grad_w[_i].detach().numpy(), w.grad[_i].numpy(), **allclose_kwargs))


if __name__ == '__main__':
    #grad_check_wo_l2_norm()
    #grad_check_with_l2_norm()
    grad_check_with_l2_norm_with_bs()

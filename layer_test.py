import time
import unittest

import paddle
import paddle.nn.functional as F
import torch


class ConvTbcTest(unittest.TestCase):
    def test_conv_tbc(self):
        x = torch.ones([1000, 64, 3])
        # x.shape = [1000, 64, 3]; (L_{in}, N, C_{in})
        weight = torch.nn.Parameter(torch.Tensor(3, 3, 64))
        # weight.shape = [3, 3, 64]; (L_f, C_{in}, C_{out})
        bias = torch.nn.Parameter(torch.Tensor(64))
        # bias.shape = [64]; (C_{out})
        conv_tbc = torch.conv_tbc(x, weight, bias)
        # conv_tbc.shape = [998, 64, 64]; (L_{out}, N, C_{out})
        conv_tbc = paddle.to_tensor(conv_tbc.detach().numpy())
        self.assertListEqual(conv_tbc.shape, [998, 64, 64])

        x = paddle.to_tensor(x.numpy())
        weight = paddle.to_tensor(weight.detach().numpy())
        bias = paddle.to_tensor(bias.detach().numpy())

        x = x.transpose((1, 2, 0))
        # x.shape = [64, 3, 1000]; (N, C_{in}, L_{in})
        weight = weight.transpose((2, 0, 1))
        # weight.shape = [64, 3, 3]; (C_{out}, C_{in}, L_f)
        conv1d = F.conv1d(x, weight, bias)
        # conv1d.shape = [64, 64, 998]; (N, C_{out}, L_{out})
        # where L_{out} = L_{in} - L_f + 1
        conv1d = conv1d.transpose((2, 0, 1))
        # conv1d.shape = [998, 64, 64]; (L_{out}, N, C_{out})
        self.assertListEqual(conv1d.shape, [998, 64, 64])

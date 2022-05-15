import unittest

import numpy as np
import paddle
import torch
import fairseq

import util


class UtilTest(unittest.TestCase):
    def test_extend_conv_spec(self):
        convolutions = [(512, 3), (512, 3, 2)]
        self.assertEqual(
            util.extend_conv_spec(convolutions), ((512, 3, 1), (512, 3, 2)))


class LayerTest(unittest.TestCase):
    def test_embedding_layer(self):
        embedding = util.Embedding(
            num_embeddings=10, embedding_dim=512, padding_idx=0)
        self.assertEqual(embedding.weight.shape, [10, 512])
        self.assertEqual(embedding.weight[0].norm(), 0)
        self.assertNotEqual(embedding.weight[1].norm(), 0)

        x_data = np.arange(3).reshape((3, 1)).astype(np.int64)
        x = paddle.to_tensor(x_data, stop_gradient=False)
        self.assertEqual(x.shape, [3, 1])

        embedding_out = embedding(x)
        self.assertEqual(embedding_out.shape, [3, 1, 512])

    def test_linear_layer(self):
        linear = util.Linear(in_features=512, out_features=256)
        # in_features, out_features
        self.assertEqual(linear.weight.shape, [512, 256])
        self.assertEqual(linear.bias.shape, [256])
        self.assertEqual(linear.bias.norm(), 0)

        x = paddle.randn((3, 512), dtype="float32")
        self.assertEqual(x.shape, [3, 512])

        y = linear(x)
        self.assertEqual(y.shape, [3, 256])

        # test pytorch linear weight shape
        torch_linear = torch.nn.Linear(in_features=512, out_features=256)
        # NOTE the following shape is different from paddle.nn.Linear.weight.shape
        # out_features, in_features
        self.assertEqual(torch_linear.weight.shape, (256, 512))

    def test_conv1d_layer(self):
        conv1d = util.Conv1D(in_channels=512, out_channels=256, kernel_size=3)
        # out_channels, in_channels, kernel_size
        self.assertEqual(conv1d.weight.shape, [256, 512, 3])
        self.assertEqual(conv1d.bias.shape, [256])
        self.assertEqual(conv1d.bias.norm(), 0)

        # batch_size, sequence_length, in_channels
        x = paddle.randn((3, 100, 512), dtype="float32")
        self.assertEqual(x.shape, [3, 100, 512])

        y = conv1d(x)
        # batch_size, sequence_length - kernel_size + 1, out_channels
        self.assertEqual(y.shape, [3, 98, 256])

        # test pytorch ConvTBC weight shape
        fairseq_conv_tbc = fairseq.modules.ConvTBC(
            in_channels=512, out_channels=256, kernel_size=3)
        # NOTE the following shape is different from paddle.nn.Conv1D.weight.shape
        # kernel_size, in_channels, out_channels
        self.assertEqual(fairseq_conv_tbc.weight.shape, (3, 512, 256))

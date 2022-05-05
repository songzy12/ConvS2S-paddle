import unittest

import numpy as np
import paddle

import util


class UtilTest(unittest.TestCase):
    def test_extend_conv_spec(self):
        convolutions = [(512, 3), (512, 3, 2)]
        self.assertEqual(util.extend_conv_spec(
            convolutions), ((512, 3, 1), (512, 3, 2)))


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
        for i in range(3):
            self.assertEqual(
                embedding_out[i][0].norm(), embedding.weight[i].norm())

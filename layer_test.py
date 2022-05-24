import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
import torch

import layer


class Conv1dTest(unittest.TestCase):

    def test_conv_tbc(self):
        x = torch.ones([1000, 64, 512])
        # x.shape = [1000, 64, 512]; (L_{in}, N, C_{in})
        weight = torch.nn.Parameter(torch.Tensor(3, 512, 768))
        # weight.shape = [3, 512, 768]; (L_f, C_{in}, C_{out})
        bias = torch.nn.Parameter(torch.Tensor(768))
        # bias.shape = [768]; (C_{out})
        conv_tbc = torch.conv_tbc(x, weight, bias)
        # conv_tbc.shape = [998, 64, 768]; (L_{out}, N, C_{out})
        conv_tbc = paddle.to_tensor(conv_tbc.detach().numpy())
        self.assertListEqual(conv_tbc.shape, [998, 64, 768])

        x = paddle.to_tensor(x.numpy())
        weight = paddle.to_tensor(weight.detach().numpy())
        bias = paddle.to_tensor(bias.detach().numpy())

        x = x.transpose((1, 2, 0))
        # x.shape = [64, 512, 1000]; (N, C_{in}, L_{in})
        weight = weight.transpose((2, 1, 0))
        # weight.shape = [768, 512, 3]; (C_{out}, C_{in}, L_f)
        conv1d = F.conv1d(x, weight, bias)
        # conv1d.shape = [64, 768, 998]; (N, C_{out}, L_{out})
        # where L_{out} = L_{in} - L_f + 1
        conv1d = conv1d.transpose((2, 0, 1))
        # conv1d.shape = [998, 64, 768]; (L_{out}, N, C_{out})
        self.assertListEqual(conv1d.shape, [998, 64, 768])


class FConvEncoderTest(unittest.TestCase):

    def test_fconv_encoder(self):
        encoder = layer.FConvEncoder(num_embeddings=10000,
                                     embed_dim=512,
                                     padding_idx=0,
                                     dropout=0.0)

        input_ids = np.random.randint(low=0, high=10000, size=(64, 1000))
        input_ids = paddle.to_tensor(input_ids)
        # input_ids.shape = [64, 1000]; (N, L_{in})

        encoder_output, encoder_padding_mask = encoder(input_ids)
        x, y = encoder_output
        # x.shape = [64, 1000, 512]; (N, L_{out}, C_{out})
        # y.shape = [64, 1000, 512]; (N, L_{out}, C_{out})
        self.assertListEqual(x.shape, [64, 1000, 512])
        self.assertListEqual(y.shape, [64, 1000, 512])
        # encoder_state.shape = [64, 1000]; (N, L)
        self.assertListEqual(encoder_padding_mask.shape, [64, 1000])


class AttentionLayerTest(unittest.TestCase):

    def test_attention_layer(self):
        attention_layer = layer.AttentionLayer(conv_channels=512,
                                               embed_dim=768)

        # batch, length_tgt, conv_channels
        x = paddle.randn((64, 1000, 512))
        # batch, length_tgt, embed_dim
        target_embedding = paddle.randn((64, 1000, 768))
        # encoder_a: batch, embed_dim, length_src; transposed by _split_encoder_out
        # encoder_b: batch, length_src, embed_dim
        encoder_out = [
            paddle.randn((64, 768, 1005)),
            paddle.randn((64, 1005, 768))
        ]
        encoder_padding_mask = None

        y, attn_scores = attention_layer(x, target_embedding, encoder_out,
                                         encoder_padding_mask)
        self.assertListEqual(
            y.shape, [64, 1000, 512])  # batch, length_tgt, conv_channels
        self.assertListEqual(attn_scores.shape,
                             [64, 1000, 1005])  # batch, length_tgt, length_src


class FConvDecoderTest(unittest.TestCase):

    def test_fconv_decoder(self):
        decoder = layer.FConvDecoder(num_embeddings=10000,
                                     embed_dim=512,
                                     padding_idx=0,
                                     convolutions=((512, 3), ) * 3,
                                     dropout=0.0)
        decoder.eval()

        prev_output_tokens = np.random.randint(low=0,
                                               high=10000,
                                               size=(64, 1000))
        prev_output_tokens = paddle.to_tensor(prev_output_tokens)

        encoder_out = [
            paddle.randn((64, 1000, 512)),
            paddle.randn((64, 1000, 512))
        ]
        encoder_padding_mask = None

        y, attn_scores = decoder(prev_output_tokens, encoder_out,
                                 encoder_padding_mask)
        # y.shape = [64, 1000, 10000]; (N, L_{out}, num_embeddings)
        self.assertListEqual(y.shape, [64, 1000, 10000])
        # attn_scores.shape =  (N, L_{out}, L_{in})
        self.assertListEqual(attn_scores.shape, [64, 1000, 1000])

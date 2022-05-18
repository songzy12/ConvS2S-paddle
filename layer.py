# TODO(songzy): positional embedding:
# - https://github.com/PaddlePaddle/PaddleNLP/blob/fa8fe0ef257a48b6517b01b177513391a6dfa8c2/paddlenlp/transformers/transformer/modeling.py#L786
# - https://github.com/pytorch/fairseq/blob/51478ad3a19feed51d4bc4df5416870b7cee5347/fairseq/models/fconv.py#L243
#
# For importance of position embeddings, see Section 5.4 of the ConvS2S paper.

# TODO(songzy): ConvTBC:
# - torch/nn/modules/conv.py
# - fairseq/modules/conv_tbc.py
# - explanation about conv_tbc: https://github.com/pytorch/fairseq/issues/172
# - conv1d workaround for conv_tbc: https://github.com/PaddlePaddle/Paddle/issues/35257 

# TODO(songzy): LinearizedConvolution
# - fairseq/modules/linearized_convolution.py
# - incremental decoding: https://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/
# - PaddleNLP: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_translation/seq2seq/seq2seq_attn.py#L242

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import util


class AttentionLayer(nn.Layer):
    def __init__(self, conv_channels, embed_dim):
        super(AttentionLayer, self).__init__()

        # projects from output of convolution to embedding dimension
        self.in_projection = util.Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = util.Linear(embed_dim, conv_channels)

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = paddle.bmm(x, encoder_out[0])

        # don't attend over padding
        if encoder_padding_mask is not None:
            x = (x.float().masked_fill(
                encoder_padding_mask.unsqueeze(1), float("-inf"))
                 .type_as(x))  # FP16 support: cast to float and back

        # softmax over last dim
        sz = x.shape
        x = F.softmax(paddle.reshape(x, (sz[0] * sz[1], sz[2])), axis=1)
        x = paddle.reshape(x, sz)
        attn_scores = x

        x = paddle.bmm(x, encoder_out[1])

        # scale attention output (respecting potentially different lengths)
        s = encoder_out[1].shape[1]
        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(
                dim=1, keepdim=True)  # exclude padding
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

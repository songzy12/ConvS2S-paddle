# For initialization strategy of layer weights, see:
#   Section 3.5 of https://arxiv.org/pdf/1705.03122.pdf

import math

import paddle
import paddle.nn as nn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    """Weight-normalized Embedding layer"""
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(0.0, 0.1))
    embedding = nn.Embedding(num_embeddings, embedding_dim,
                             padding_idx=padding_idx, weight_attr=weight_attr)
    # The weight of padding_idx is set to 0 when embedding layer is initialized:
    #   embedding.weight[padding_idx] = 0.0
    return embedding


def Linear(in_features, out_features, dropout=0.0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    weight_attr = paddle.ParamAttr(
        name="weight",
        initializer=nn.initializer.Normal(0.0, math.sqrt((1 - dropout) / in_features)))
    bias_attr = paddle.ParamAttr(
        name="bias",
        initializer=nn.initializer.Constant(value=0.0))

    m = nn.Linear(in_features, out_features,
                  weight_attr=weight_attr, bias_attr=bias_attr)
    return nn.utils.weight_norm(m)

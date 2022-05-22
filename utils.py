# For initialization strategy of layer weights, see:
#   Section 3.5 of https://arxiv.org/pdf/1705.03122.pdf
# For weight normalization, see:
#   Section 4.2 of https://arxiv.org/pdf/1705.03122.pdf, then
#   https://arxiv.org/pdf/1602.07868.pdf

import math

import paddle
import paddle.nn as nn


def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1, ))
        else:
            raise Exception("invalid number of parameters in convolution spec "
                            + str(spec) + ". expected 2 or 3")
    return tuple(extended)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    """Weight-normalized Embedding layer"""
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
        mean=0.0, std=0.1))
    embedding = nn.Embedding(
        num_embeddings,
        embedding_dim,
        padding_idx=padding_idx,
        weight_attr=weight_attr)
    # The weight of padding_idx is set to 0 when embedding layer is initialized:
    #   embedding.weight[padding_idx] = 0.0
    return embedding


def Linear(in_features, out_features, dropout=0.0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    std = math.sqrt((1 - dropout) / in_features)
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
        mean=0.0, std=std))
    bias_attr = paddle.ParamAttr(
        initializer=nn.initializer.Constant(value=0.0))

    m = nn.Linear(
        in_features,
        out_features,
        weight_attr=weight_attr,
        bias_attr=bias_attr)
    # paddle/nn/functional/common.py
    #   If the weight is a 2-D tensor of shape :math:`[in\_features, out\_features]`
    # torch/nn/modules/linear.py
    #   weight: the learnable weights of the module of shape :math:`(\text{out\_features}, \text{in\_features})`.
    return nn.utils.weight_norm(m, dim=1)


def Conv1D(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    """Weight-normalized Conv1D layer"""
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
        mean=0.0, std=std))
    bias_attr = paddle.ParamAttr(
        initializer=nn.initializer.Constant(value=0.0))

    m = nn.Conv1D(
        in_channels,
        out_channels,
        kernel_size,
        data_format="NLC",
        weight_attr=weight_attr,
        bias_attr=bias_attr,
        **kwargs)

    # paddle/nn/layer/conv.py
    #   weight: 3-D tensor with shape: (out_channels, in_channels, kernel_size)
    # fairseq/modules/conv_tbc.py
    # torch/nn/functional.py
    #   weight: filter of shape (:math:`\text{kernel width} \times \text{in\_channels} \times \text{out\_channels}`)
    return nn.utils.weight_norm(m, dim=0)

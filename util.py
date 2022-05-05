# For initialization strategy of layer weights, see:
#   Section 3.5 of https://arxiv.org/pdf/1705.03122.pdf
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
            extended.append(spec + (1,))
        else:
            raise Exception(
                "invalid number of parameters in convolution spec "
                + str(spec)
                + ". expected 2 or 3"
            )
    return tuple(extended)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    """Weight-normalized Embedding layer"""
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(0.0, 0.1))
    embedding = nn.Embedding(num_embeddings, embedding_dim,
                             padding_idx=padding_idx, weight_attr=weight_attr)
    # The weight of padding_idx is set to 0 when embedding layer is initialized:
    #   embedding.weight[padding_idx] = 0.0
    return embedding

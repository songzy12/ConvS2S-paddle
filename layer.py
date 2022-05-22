# TODO(songzy): LinearizedConvolution
# - fairseq/modules/linearized_convolution.py
# - incremental decoding: https://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/
# - PaddleNLP: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_translation/seq2seq/seq2seq_attn.py#L242

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import utils


class FConvEncoder(nn.Layer):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.
    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int, optional): embedding dimension
        padding_idx (int, optional): padding index for :obj:`Embedding`
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        padding_idx=None,
        convolutions=((512, 3), ) * 20,
        dropout=0.1,
    ):
        super().__init__()
        self.dropout_module = nn.Dropout(p=dropout,
                                         name=self.__class__.__name__)
        self.num_attention_layers = None

        self.padding_idx = padding_idx
        self.embed_tokens = utils.Embedding(vocab_size, embed_dim,
                                            self.padding_idx)

        convolutions = utils.extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = utils.Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = []
        self.convolutions = []
        self.residuals = []

        layer_in_channels = [in_channels]
        for _, (out_channels, kernel_size,
                residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(
                utils.Linear(residual_dim, out_channels
                             ) if residual_dim != out_channels else None)
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                utils.Conv1D(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    dropout=dropout,
                    padding=padding,
                ))
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = utils.Linear(in_channels, embed_dim)

    def forward(self, src_tokens):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
        Returns:
            dict:
                - **encoder_out** (tuple): a tuple with two elements, where the
                  first element is the last encoder layer's output and the
                  second element is the same quantity summed with the input
                  embedding (used for attention). The shape of both tensors is
                  `(batch, src_len, embed_dim)`.
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        # TODO(songzy): positional embedding:
        # - https://github.com/PaddlePaddle/PaddleNLP/blob/fa8fe0ef257a48b6517b01b177513391a6dfa8c2/paddlenlp/transformers/transformer/modeling.py#L786
        # - https://github.com/pytorch/fairseq/blob/51478ad3a19feed51d4bc4df5416870b7cee5347/fairseq/models/fconv.py#L243
        #
        # For importance of position embeddings, see Section 5.4 of the ConvS2S paper.
        x = self.embed_tokens(src_tokens)
        x = self.dropout_module(x)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # used to mask padding in input
        encoder_padding_mask = src_tokens.equal(self.padding_idx)  # -> B x T
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(self.projections, self.convolutions,
                                         self.residuals):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            if encoder_padding_mask is not None:
                x = paddle.where(encoder_padding_mask.unsqueeze(-1),
                                 paddle.full(x.shape, 0, x.dtype), x)

            x = self.dropout_module(x)
            if conv._kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv._kernel_size[0] - 1) // 2
                padding_r = conv._kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            # x.shape: B x T x C
            x = F.glu(x, axis=2)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # project back to size of embedding
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            x = paddle.where(encoder_padding_mask.unsqueeze(-1),
                             paddle.full(x.shape, 0, x.dtype), x)

        # TODO(songzy): scale gradients (this only affects backward, not forward)
        # x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        # encoder_out, encoder_padding_mask
        return (x, y), encoder_padding_mask


class AttentionLayer(nn.Layer):

    def __init__(self, conv_channels, embed_dim):
        super(AttentionLayer, self).__init__()

        # projects from output of convolution to embedding dimension
        self.in_projection = utils.Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = utils.Linear(embed_dim, conv_channels)

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = paddle.bmm(x, encoder_out[0])

        # don't attend over padding
        if encoder_padding_mask is not None:
            x = paddle.where(
                encoder_padding_mask.unsqueeze(1),
                paddle.full(x.shape, float("-inf"),
                            x.float().dtype),
                x.float()).type_as(x)  # FP16 support: cast to float and back

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

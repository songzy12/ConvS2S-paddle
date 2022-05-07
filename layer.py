# TODO(songzy): take a look at the positional embedding:
# - https://github.com/PaddlePaddle/PaddleNLP/blob/fa8fe0ef257a48b6517b01b177513391a6dfa8c2/paddlenlp/transformers/transformer/modeling.py#L786
# - https://github.com/pytorch/fairseq/blob/51478ad3a19feed51d4bc4df5416870b7cee5347/fairseq/models/fconv.py#L243
#
# For effectiveness of position embeddings, see Section 5.4 of the ConvS2S paper.

# TODO(songzy): take a look at ConvTBC and LinearizedConvolution:
# - torch/nn/modules/conv.py
# - fairseq/modules/linearized_convolution.py
# - fairseq/modules/conv_tbc.py
# - https://github.com/PaddlePaddle/Paddle/issues/35257
# - https://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/
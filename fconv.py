import paddle.nn as nn

import layer


class FConvModel(nn.Layer):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            embed_dim,
            convolutions=((512, 3), ) * 20,
            padding_idx=0,
            dropout=0., ):
        super(FConvModel, self).__init__()

        self.encoder = layer.FConvEncoder(src_vocab_size, embed_dim,
                                          padding_idx, convolutions, dropout)
        self.decoder = layer.FConvDecoder(tgt_vocab_size, embed_dim,
                                          padding_idx, convolutions, dropout)

    def forward(self, src_tokens, prev_output_tokens):
        encoder_out, encoder_padding_mask = self.encoder(src_tokens)
        decoder_out, _ = self.decoder(
            prev_output_tokens, encoder_out, encoder_padding_mask)
        return decoder_out

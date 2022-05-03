https://github.com/pytorch/fairseq/blob/main/fairseq/models/fconv.py

```
class FConvDecoder(FairseqIncrementalDecoder):
    """Convolutional decoder"""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        embed_dict=None,
        out_embed_dim=256,
        max_positions=1024,
        convolutions=((512, 3),) * 20,
        attention=True,
        dropout=0.1,
        share_embed=False,
        positional_embeddings=True,
        adaptive_softmax_cutoff=None,
        adaptive_softmax_dropout=0.0,
    ):
        self.attention = nn.ModuleList()
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            self.attention.append(
                AttentionLayer(out_channels, embed_dim) if attention[i] else None
            )

        if adaptive_softmax_cutoff is not None:
        else:
            self.fc2 = Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, (
                    "Shared embed weights implies same dimensions "
                    " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
                )
                self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):

        for proj, conv, attention, res_layer in zip(
            self.projections, self.convolutions, self.attention, self.residuals
        ):
            # attention
            if attention is not None:
                x = self._transpose_if_training(x, incremental_state)

                x, attn_scores = attention(
                    x, target_embedding, (encoder_a, encoder_b), encoder_padding_mask
                )

                if not self.training and self.need_attn:
                    attn_scores = attn_scores / num_attn_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)

                x = self._transpose_if_training(x, incremental_state)
```
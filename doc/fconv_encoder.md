https://github.com/pytorch/fairseq/blob/main/fairseq/models/fconv.py

```
def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
```

```
class FConvEncoder(FairseqEncoder):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.
    Args:
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_dim (int, optional): embedding dimension
        embed_dict (str, optional): filename from which to load pre-trained
            embeddings
        max_positions (int, optional): maximum supported input sequence length
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
    """

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        embed_dict=None,
        max_positions=1024,
        convolutions=((512, 3),) * 20,
        dropout=0.1,
    ):
        in_channels = convolutions[0][0]
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for _, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(
                Linear(residual_dim, out_channels)
                if residual_dim != out_channels
                else None
            )
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                ConvTBC(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    dropout=dropout,
                    padding=padding,
                )
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
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
        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(
            self.projections, self.convolutions, self.residuals
        ):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            x = self.dropout_module(x)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=2)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)
```
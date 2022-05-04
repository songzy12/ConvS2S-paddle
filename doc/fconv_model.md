## FConvModel

fairseq/models/fconv.py

`FConvModel` is a subclass of `FairseqEncoderDecoderModel`, which means it has a `encoder` and a `decoder`.

## FConvEncoder

`FConvEncoder` is a subclass of `FairseqEncoder`.

### ConvTBC

TBC: time x batch x channel.

Why TBC: https://github.com/pytorch/fairseq/issues/525

> for the original convolutional model we found TBC results in fewer transpose/contiguous calls (esp. relating to the attention mechanism) and was therefore faster.

## FConvDecoder

`FConvDecoder` is a subclass of `FairseqIncrementalDecoder`.

### LinearizedConv1d

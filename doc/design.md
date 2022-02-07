## FConvModel

fairseq/models/fconv.py

`FConvModel` is a subclass of `FairseqEncoderDecoderModel`, which means it has a `encoder` and a `decoder`.

## FConvEncoder

`FConvEncoder` is a subclass of `FairseqEncoder`.

### ConvTBC

TBC: time x batch x channel.

## FConvDecoder

`FConvDecoder` is a subclass of `FairseqIncrementalDecoder`.

### LinearizedConv1d

## FairSeq Cli

- fairseq-preprocess
- fairseq-train
- fairseq-generate

`docs/command_line_tools.rst`

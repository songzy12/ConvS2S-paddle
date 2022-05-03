
## FairseqEncoderDecoderModel

`fairseq/models/fairseq_model.py`
```
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out
```

## FairseqEncoder

`fairseq/models/fairseq_encoder.py`

## FairseqIncrementalDecoder 

`fairseq/models/fairseq_incremental_decoder.py`

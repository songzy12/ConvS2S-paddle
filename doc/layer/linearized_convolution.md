<https://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>
  
`fairseq/modules/linearized_convolution.py`
```
@with_incremental_state
class LinearizedConvolution(ConvTBC):
    """An optimized version of nn.Conv1d.

    At training time, this module uses ConvTBC, which is an optimized version
    of Conv1d. At inference time, it optimizes incremental generation (i.e.,
    one time step at a time) by replacing the convolutions with linear layers.
    Note that the input order changes from training to inference.
    """
    def forward(self, input, incremental_state=None):
        """
        Args:
            incremental_state: Used to buffer signal; if not None, then input is
                expected to contain a single frame. If the input order changes
                between time steps, call reorder_incremental_state.

        Input:
            Time x Batch x Channel during training
            Batch x Time x Channel during inference
        """
        if incremental_state is None:
            output = super().forward(input)

        # reshape weight
        weight = self._get_linearized_weight()
        with torch.no_grad():
            output = F.linear(input.view(bsz, -1), weight, self.bias)
```

In context of NLP, "a single frame" means "a single word".

## linearized weight

```
    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = torch.nn.Parameter(
                weight.view(self.out_channels, -1)
            )
        return self._linearized_weight
```

## incremental state

```
class FConvDecoder(FairseqIncrementalDecoder):
    def forward(self,
                prev_output_tokens,
                encoder_out=None,
                incremental_state=None,
                **unused):
        for proj, conv, attention, res_layer in zip(
                self.projections, self.convolutions, self.attention,
                self.residuals):

            x = conv(x, incremental_state)
```

So when would this incremental_state be set?

`examples/translation/README.md`
```
# Evaluate
fairseq-generate \
    data-bin/fconv_wmt_en_fr \
    --path checkpoints/fconv_wmt_en_fr/checkpoint_best.pt \
    --beam 5 --remove-bpe
```

`fairseq_cli/generate.py`
```
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    for sample in progress:
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
    
            # Process top predictions
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
            
```

`fairseq/tasks/fairseq_task.py`
```
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
                seq_gen_cls = SequenceGenerator
```

`fairseq/sequence_generator.py`
```
class SequenceGenerator(nn.Module):
    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad

        for step in range(max_len + 1):  # one extra step for EOS marker
            with torch.autograd.profiler.record_function(
                "EnsembleModel: forward_decoder"
            ):
                lprobs, avg_attn_scores = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                )

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            if eos_bbsz_idx.numel() > 0:
                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
```

`fairseq/models/fconv.py`
```
class FConvDecoder(FairseqIncrementalDecoder):
    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        if encoder_out is not None:
            encoder_a, encoder_b = self._split_encoder_out(
                encoder_out, incremental_state
            )

    def _split_encoder_out(self, encoder_out, incremental_state):
        result = (encoder_a, encoder_b)
        if incremental_state is not None:
            utils.set_incremental_state(self, incremental_state, "encoder_out", result)
```

On the other hand, in PaddleNLP, `examples/machine_translation/seq2seq/train.py`:

```
class Seq2SeqAttnInferModel(Seq2SeqAttnModel):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout_prob=0.,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256):
        # Dynamic decoder for inference
        self.beam_search_decoder = nn.BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedder,
            output_fn=self.decoder.output_layer)

    def forward(self, src, src_length):
        # Dynamic decoding with beam search
        seq_output, _ = nn.dynamic_decode(
            decoder=self.beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=self.max_out_len,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask)
```

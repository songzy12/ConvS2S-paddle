`torch/nn/modules/sparse.py`

```
class Embedding(Module):
    Shape:
        - Input: :math:`(*)`, IntTensor or LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
```

Note: the shape of the Embedding output can not be passed to the Conv1d directly.
- output shape of Embedding: B x T x C 
- input shape of Conv1d: B x C x T

Here we have 
- B: batch size
- T: time, i.e, sentence length
- C: channel, i.e., word embedding dimension

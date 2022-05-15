What is the difference of CONV_TBC and CONV1D?

`torch/_C/_VariableFunctions.pyi`

```
@overload
def conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[_int, _size]=1, padding: Union[_int, _size]=0, dilation: Union[_int, _size]=1, groups: _int=1) -> Tensor: ...
@overload
def conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[_int, _size]=1, padding: str="valid", dilation: Union[_int, _size]=1, groups: _int=1) -> Tensor: ...

def conv_tbc(input: Tensor, weight: Tensor, bias: Tensor, pad: _int=0) -> Tensor: ...
```

`torch/nn/modules/conv.py`

```
class Conv1D(_ConvNd):
    """
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where
          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
    """
    def forward(self, x):
        out = F.conv1d(
            x,
            self.weight,
            bias=self.bias,
            padding=padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format)
```

`fairseq/modules/conv_tbc.py`

```
class ConvTBC(torch.nn.Module):
    def forward(self, input):
        return torch.conv_tbc(
            input.contiguous(), self.weight, self.bias, self.padding[0]
        )
```


`torch/nn/functional.py`
`torch/_C/_VariableFunctions.pyi`

```
@overload
def conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[_int, _size]=1, padding: Union[_int, _size]=0, dilation: Union[_int, _size]=1, groups: _int=1) -> Tensor: ...
@overload
def conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[_int, _size]=1, padding: str="valid", dilation: Union[_int, _size]=1, groups: _int=1) -> Tensor: ...

def conv_tbc(input: Tensor, weight: Tensor, bias: Tensor, pad: _int=0) -> Tensor: ...
```

`aten/src/ATen/native/Convolution.cpp`
```
    output = at::convolution(input, weight, bias, stride, padding, dilation, false, {0}, groups);
```

`aten/src/ATen/native/ConvolutionTBC.cpp`
```
  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = weight_size[2];
  auto kw = weight_size[0];
  auto olen = input_size[0] - kw + 1 + pad * 2;
  auto real_pad = (olen - ilen + kw - 1) / 2;

  // Make sure shapes are correct.
  // Input = (time, batch, in_channels)
  // Weight = (kernel_width, in_channels, out_channels)
  // Bias = (out_channels)
  TORCH_CHECK(inputPlanes == weight_size[1], "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  TORCH_CHECK(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");
      
  // input * weights + bias -> output_features
  Tensor output = at::empty({
    olen,
    input_size[1],
    weight_size[2],
  }, self.options());
      O.addmm_(I, W);
```

An explanation of the above terms in context of NLP:
- time: the number of words in a sentence
- batch: the number of sentences in a batch
- in_channel: the dimension of word embeddings
- kenel_width: the width of convolution kernels, i.e., the number of words involved in a single convolution computation
- out_channel: the number of convolution kernels


`paddle/nn/layer/conv.py`

```
class Conv1D(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCL"):
```

`paddle/nn/functional/conv.py`
```
def conv1d(x,
           weight,
           bias=None,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           data_format='NCL',
           name=None):
    r"""
          Input shape: :math:`(N, C_{in}, L_{in})`
          Filter shape: :math:`(C_{out}, C_{in}, L_f)`
          Output shape: :math:`(N, C_{out}, L_{out})`
            where
            L_{out} = \frac{(L_{in} + 2 * padding - (dilation * (L_f - 1) + 1))}{stride} + 1
    """
```

NOTE: to take input from Embedding layer directly, we should set data_format as `NLC`.

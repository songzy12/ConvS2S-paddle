import unittest

import numpy as np
import paddle

import fconv


class FConvModelTest(unittest.TestCase):
    def test_fconv_model(self):
        model = fconv.FConvModel(
            src_vocab_size=10000,
            tgt_vocab_size=10000,
            embed_dim=512,
            convolutions=((512, 3), ) * 2,
            padding_idx=0,
            dropout=0.0)
        # Set model to eval mode so we can have avg_attn_scores.    
        model.eval()

        src_tokens = np.random.randint(low=0, high=10000, size=(64, 1000))
        src_tokens = paddle.to_tensor(src_tokens)
        prev_output_tokens = np.random.randint(
            low=0, high=10000, size=(64, 1000))
        prev_output_tokens = paddle.to_tensor(prev_output_tokens)

        decoder_output = model(src_tokens, prev_output_tokens)

        self.assertListEqual(decoder_output.shape, [64, 1000, 10000])

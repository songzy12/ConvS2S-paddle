## FairSeq Cli

- fairseq-preprocess
- fairseq-train
- fairseq-generate

`docs/command_line_tools.rst`, `setup.py`

```
    setup(
        entry_points={
            "console_scripts": [
                "fairseq-eval-lm = fairseq_cli.eval_lm:cli_main",
```

`examples/translation/README.md`

## Preprocess

### Command

```
cd examples/translation/
bash prepare-wmt14en2fr.sh
cd ../..
```

```
TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
```

### Code

`fairseq/binarizer.py`

```
class FileBinarizer:
    def _binarize_file_chunk(
                ds.add_item(binarizer.binarize_line(line, summary))

class VocabularyDatasetBinarizer(Binarizer):
    def binarize_line(
            ids = torch.IntTensor(id_list)
```

### Result

```
✗ tree data-bin/wmt17_en_de
data-bin/wmt17_en_de
├── dict.de.txt
├── dict.en.txt
├── preprocess.log
├── test.en-de.de.bin
├── test.en-de.de.idx
├── test.en-de.en.bin
├── test.en-de.en.idx
├── train.en-de.de.bin
├── train.en-de.de.idx
├── train.en-de.en.bin
├── train.en-de.en.idx
├── valid.en-de.de.bin
├── valid.en-de.de.idx
├── valid.en-de.en.bin
└── valid.en-de.en.idx

0 directories, 15 files
```

## Train

### Command

```
mkdir -p checkpoints/fconv_wmt_en_de
fairseq-train \
    data-bin/wmt17_en_de \
    --arch fconv_wmt_en_de \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000 \
    --save-dir checkpoints/fconv_wmt_en_de
```

### Code

### Result

## Generate

```
fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe
```
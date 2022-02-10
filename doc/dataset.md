`examples/translation/README.md`

```
cd examples/translation/
bash prepare-wmt14en2de.sh --icml17
```

```
URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
)
```

## Moses

```
git clone https://github.com/moses-smt/mosesdecoder.git
```

```
SCRIPTS=mosesdecoder/scripts

TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
```

## BPE

https://en.wikipedia.org/wiki/Byte_pair_encoding

```
git clone https://github.com/rsennrich/subword-nmt.git
```

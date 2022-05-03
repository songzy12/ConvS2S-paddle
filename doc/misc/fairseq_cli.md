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

`fairseq_cli/train.py`

```
def main(cfg: FairseqConfig) -> None:
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)

    trainer = Trainer(cfg, task, model, criterion, quantizer)    
    while epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
```

```
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:

    for i, samples in enumerate(progress):
            log_output = trainer.train_step(samples)
```

`fairseq/trainer.py`

```
    def train_step(self, samples, raise_oom=False):
                    # forward and backward
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample=sample,
                        model=self.model,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        update_num=self.get_num_updates(),
                        ignore_grad=is_dummy_batch,
                        **extra_kwargs,
                    )
                # take an optimization step
                self.task.optimizer_step(
                    self.optimizer, model=self.model, update_num=self.get_num_updates()
                )
```

`fairseq/tasks/translation.py`
`fairseq/tasks/fairseq_task.py`

```
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):    
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()
```

### Result

## Generate

```
fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe
```
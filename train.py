from args import parse_args

import paddle
import paddle.nn as nn
from paddlenlp.metrics import Perplexity

from fconv import FConvModel
from data import create_train_loader
from loss import CrossEntropyCriterion


def do_train(args):
    device = paddle.set_device(args.device)

    # Define dataloader
    train_loader, eval_loader, src_vocab_size, tgt_vocab_size, padding_idx = create_train_loader(
        args)

    # TODO(songzy): check how to set param convolutions.
    model = paddle.Model(
        FConvModel(src_vocab_size, tgt_vocab_size, args.embed_dim, ((512, 3), )
                   * 20, padding_idx, args.dropout))

    grad_clip = nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        grad_clip=grad_clip)

    ppl_metric = Perplexity()
    model.prepare(optimizer, CrossEntropyCriterion(), ppl_metric)

    print(args)
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    benchmark_logger = paddle.callbacks.ProgBarLogger(
        log_freq=args.log_freq, verbose=3)

    model.fit(train_data=train_loader,
              eval_data=eval_loader,
              epochs=args.max_epoch,
              eval_freq=1,
              save_freq=1,
              save_dir=args.model_path,
              callbacks=[benchmark_logger])


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
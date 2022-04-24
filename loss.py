import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CrossEntropyCriterion(nn.Layer):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, predict, label, trg_mask):
        cost = F.cross_entropy(
            input=predict, label=label, soft_label=False, reduction='none')
        cost = paddle.squeeze(cost, axis=[2])
        masked_cost = cost * trg_mask
        batch_mean_cost = paddle.mean(masked_cost, axis=[0])
        seq_cost = paddle.sum(batch_mean_cost)

        return seq_cost

import torch
import torch.nn.functional as F
from transformers import Trainer

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss

class RDropTrainer(Trainer):
    def __init__(self, *args, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # R-Drop 损失的权重

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs1 = model(**inputs)
        loss1 = outputs1.loss
        logits1 = outputs1.logits

        outputs2 = model(**inputs)
        loss2 = outputs2.loss
        logits2 = outputs2.logits

        kl_loss = compute_kl_loss(logits1, logits2)
        loss = (loss1 + loss2) / 2 + self.alpha * kl_loss

        return (loss, outputs1) if return_outputs else loss

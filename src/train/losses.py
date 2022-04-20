import torch


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100, position_weight=None):
    """From fairseq
    params target: [batch_size, ln]
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)  # [bsz, ln, 1]
    smooth_loss = -lprobs.mean(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        pass
    nll_loss = nll_loss.squeeze(-1)  # [bsz, ln]
    smooth_loss = smooth_loss.squeeze(-1)  # [bsz, ln]

    loss = (1.0 - epsilon) * nll_loss + epsilon * smooth_loss

    batch_size, length, _ = target.size()
    if position_weight is not None:
        position_penalty = torch.linspace(position_weight, 1.0, length).repeat([batch_size, 1]).to(target.device)
        # print(f"position_penalty: {position_penalty.size()}")
        # print(f"loss: {loss.size()}")
        loss = torch.mul(loss, position_penalty)
        nll_loss = torch.mul(nll_loss, position_penalty)
    else:
        pass
    return loss, nll_loss


def cross_entropy_loss(logits, target, ignore_index=-100):
    """
    logits: raw logits [bsz, C]
    target: [bsz]
    """
    loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    output = loss(input=logits, target=target)
    return output


def kld_loss(logits, target):
    """
    @param logits: [B, vocab_size]
    @param target: [B, vocab_size]
    @return:
    """
    loss = torch.nn.KLDivLoss()
    output = loss(logits, target)
    return output


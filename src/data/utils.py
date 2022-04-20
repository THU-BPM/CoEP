import torch


def collate_fn_test(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """

    all_input_ids, all_input_mask, all_segment_ids, all_input_lens = map(torch.stack, zip(*batch))
    max_len = max(all_input_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]

    return all_input_ids, all_input_mask, all_segment_ids


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """

    all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_lens, all_labels_mask, all_labels_len, all_cls_label = map(
        torch.stack, zip(*batch))
    max_len = max(all_input_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    max_label_len = max(all_labels_len).item()
    # print(f"max_input_len: {max_len}, max_label_len: {max_label_len}")
    all_label_ids = all_label_ids[:, :max_label_len]
    all_labels_mask = all_labels_mask[:, :max_label_len]
    all_cls_label = all_cls_label

    # input_ids, input_mask, segment_ids, label_ids, labels_masks, labels_segment_ids, cls_label
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_labels_mask, all_cls_label


def collate_fn_event(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_lens, \
    all_labels_mask, all_labels_len, all_label_segment_ids, all_label = map(torch.stack, zip(*batch))
    max_len = max(all_input_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    max_label_len = max(all_labels_len).item()
    # print(f"max_input_len: {max_len}, max_label_len: {max_label_len}")
    all_label_ids = all_label_ids[:, :max_label_len]
    all_labels_mask = all_labels_mask[:, :max_label_len]
    all_label_segment_ids = all_label_segment_ids[:, :max_label_len]
    all_label = all_label
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_labels_mask, all_label_segment_ids, all_label


def collate_fn_context(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_lens, \
    all_labels_mask, all_labels_len = map(torch.stack, zip(*batch))
    all_input_ids = all_input_ids
    all_input_mask = all_input_mask
    all_segment_ids = all_segment_ids
    # print(f"max_input_len: {max_len}, max_label_len: {max_label_len}")
    all_label_ids = all_label_ids
    all_labels_mask = all_labels_mask
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_labels_mask


def collate_fn_ending(batch):
    all_input_ids, all_input_mask, all_segment_ids, all_input_lens, all_next_ids, all_next_mask, all_labels_len, all_label = map(
        torch.stack, zip(*batch))
    all_input_ids = all_input_ids
    all_input_mask = all_input_mask
    all_segment_ids = all_segment_ids
    # print(f"max_input_len: {max_len}, max_label_len: {max_label_len}")
    all_next_ids = all_next_ids
    all_next_mask = all_next_mask
    all_label = all_label
    return all_input_ids, all_input_mask, all_segment_ids, all_next_ids, all_next_mask, all_label

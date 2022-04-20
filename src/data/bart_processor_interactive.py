import torch
from src.data.bart_processor_story import InputFeature
from torch.utils.data import TensorDataset


def concat_tokens_pair(tokens_a, tokens_b, right_segment_ids, max_len, pad_token_id, seg_token_id):
    """
    tokens_a ids: (ln)
    """

    a_ids = tokens_a['input_ids'][0]
    b_ids = tokens_b['input_ids'][0]
    a_len = tokens_a['length'][0]
    b_len = tokens_b['length'][0]

    left_ids = a_ids
    right_ids = b_ids
    if a_len + b_len > max_len:
        print(f"a_len: {a_len}, b_len: {b_len}, max_len: {max_len}")
        left_ids = torch.cat([left_ids[:max_len - 1 - b_len], torch.tensor([seg_token_id], dtype=left_ids.dtype)])
        input_ids = torch.cat([left_ids, right_ids[:b_len]])
        if right_segment_ids is None:
            segment_ids = torch.cat([torch.zeros([max_len - b_len], dtype=torch.int),
                                     torch.ones([b_len], dtype=torch.int)])
        else:
            segment_ids = torch.cat([torch.zeros([max_len - b_len], dtype=torch.int),
                                     1 + right_segment_ids[:b_len]])
        input_masks = torch.ones([max_len], dtype=torch.int)
        input_lens = max_len
    else:
        input_ids = torch.cat([left_ids[:a_len], right_ids[:b_len]])
        input_len = a_len + b_len
        input_ids = torch.cat(
            [input_ids, pad_token_id * torch.ones([max_len - input_len], dtype=torch.int)])
        if right_segment_ids is None:
            segment_ids = torch.cat([torch.zeros([a_len], dtype=torch.int),
                                     torch.ones([b_len], dtype=torch.int),
                                     torch.zeros([max_len - input_len], dtype=torch.int)])
        else:
            segment_ids = torch.cat([torch.zeros([a_len], dtype=torch.int),
                                     1 + right_segment_ids[:b_len],
                                     torch.zeros([max_len - input_len], dtype=torch.int)])
        input_masks = torch.cat([torch.ones([input_len], dtype=torch.int),
                                 torch.zeros([max_len - input_len], dtype=torch.int)])
        input_lens = input_len
    return {"input_ids": [input_ids],
            "segment_ids": [segment_ids],
            "attention_masks": [input_masks],
            "length": [input_lens]}


def create_dense_feature(input_content, tokenizer, max_seq_len, his_len):
    """
    type of following variables are string
    """
    context = input_content['context']
    event = input_content['event']

    tokens_his = tokenizer([context], padding=True, return_length=True, return_tensors='pt')
    tokens_a = tokenizer([event], padding=True, return_length=True, return_tensors='pt')  # event sentence
    tokens_context = concat_tokens_pair(tokens_his,
                                        tokens_a,
                                        right_segment_ids=None,
                                        max_len=his_len + max_seq_len,
                                        pad_token_id=tokenizer.pad_token_id,
                                        seg_token_id=tokenizer.eos_token_id)

    # the type of following variables are list, flat them
    input_ids = tokens_context["input_ids"][0]
    input_masks = tokens_context["attention_masks"][0]
    segment_ids = tokens_context["segment_ids"][0]
    input_lengths = tokens_context["length"][0]

    return {"input_ids": input_ids,
            "input_masks": input_masks,
            "segment_ids": segment_ids,
            "input_lengths": input_lengths}


def create_test_features(sentences, tokenizer, max_seq_len, his_len):
    """
    sentences: list of list of strings
    """
    features = []
    for sentence in sentences:
        assert type(sentence) == list
        if len(sentence) == 1:
            context = "<mask>"
        else:
            context = '|'.join(sentence[:-1])

        input_content = {'context': context,
                         'event': sentence[-1]}

        feature_dict = create_dense_feature(input_content, tokenizer, max_seq_len, his_len)
        feature = InputFeature(input_ids=feature_dict['input_ids'],
                               input_mask=feature_dict['input_masks'],
                               segment_ids=feature_dict['segment_ids'],
                               input_len=feature_dict['input_lengths'],
                               label_ids=None,
                               labels_mask=None,
                               labels_len=None,
                               cls_label=None)
        features.append(feature)
    return features


def create_test_dataset(sentences, tokenizer, max_seq_len, his_len, is_sorted=False):
    # Convert to Tensors and build dataset
    features = create_test_features(sentences, tokenizer, max_seq_len, his_len)
    if is_sorted:
        features = sorted(features, key=lambda x: x.input_len, reverse=True)
    all_input_ids = torch.cat([f.input_ids.unsqueeze(0) for f in features], dim=0)
    all_input_mask = torch.cat([f.input_mask.unsqueeze(0) for f in features], dim=0)
    all_segment_ids = torch.cat([f.segment_ids.unsqueeze(0) for f in features], dim=0)
    all_input_lens = torch.cat([f.input_len.unsqueeze(0) for f in features], dim=0)
    dataset = TensorDataset(all_input_ids,
                            all_input_mask,
                            all_segment_ids,
                            all_input_lens)
    return dataset

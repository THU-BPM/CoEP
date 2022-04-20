

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, target=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.target = target


class InputFeature(object):
    '''
    A single set of features of data.
    '''

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, input_len, labels_mask, labels_len, cls_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len
        self.labels_mask = labels_mask
        self.labels_len = labels_len
        self.cls_label = cls_label

    def value(self):
        result = []
        result += [self.input_ids]
        result += [self.labels_mask]
        result += [self.labels_len]
        result += [self.cls_label]
        return result
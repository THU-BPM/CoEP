r"""Functional interface"""
from typing import List, Dict
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import sentence_bleu

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification_report

__call__ = ['Accuracy', 'AUC', 'F1Score', 'EntityScore', 'ClassReport', 'MultiLabelReport', 'AccuracyThresh']

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class Accuracy(Metric):
    """
        >>> metric = Accuracy(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    """

    def __init__(self, topK, ignore_index=-100):
        super(Accuracy, self).__init__()
        self.topK = topK
        self.ignore_index = ignore_index
        self.reset()

    def __call__(self, logits, target, ignore_mask=None):
        """
        :param logits: Tensor [num, vocab_size]
        :param target: Tesnor [num]
        :param ignore_mask : Tensor [num]
        :return:
        """
        _, pred = logits.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        if ignore_mask is None:
            ignore_mask = target.eq(self.ignore_index).expand_as(pred)
        else:
            ignore_mask = ignore_mask.eq(1).expand_as(pred)
        masked_num = target.eq(self.ignore_index).sum()
        masked_correct = correct.masked_fill(ignore_mask, False)
        self.correct_k = masked_correct[:self.topK].contiguous().view(-1).float().sum(0)
        self.total = target.size(0) - masked_num

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k) / self.total

    def name(self):
        return 'accuracy'


class RECALL(Metric):
    """
        >>> metric = RECALL(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    """

    def __init__(self, topK, ignore_index=-100):
        super(RECALL, self).__init__()
        self.topK = topK
        self.ignore_index = ignore_index
        self.reset()

    def __call__(self, logits, target):
        """
        :param logits: Tensor [num, vocab_size]
        :param target: Tesnor [num]
        :return:
        """
        _, pred = logits.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        ignore_mask = target.eq(self.ignore_index).expand_as(pred)
        masked_num = target.eq(self.ignore_index).sum()
        masked_correct = correct.mask_fill(ignore_mask, False)
        self.correct_k = masked_correct[:self.topK].contiguous().view(-1).float().sum(0)
        self.total = target.size(0) - masked_num

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k) / self.total

    def name(self):
        return 'accuracy'


class AccuracyThresh(Metric):
    """
    Example:
        >>> metric = AccuracyThresh(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    """

    def __init__(self, thresh=0.5):
        super(AccuracyThresh, self).__init__()
        self.thresh = thresh
        self.reset()

    def __call__(self, logits, target):
        self.y_pred = logits.sigmoid()
        self.y_true = target

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        data_size = self.y_pred.size(0)
        acc = np.mean(((self.y_pred > self.thresh) == self.y_true.byte()).float().cpu().numpy(), axis=1).sum()
        return acc / data_size

    def name(self):
        return 'accuracy'


class AUC(Metric):
    '''
    AUC score
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = AUC(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self, task_type='multiclass', average='micro'):
        super(AUC, self).__init__()

        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']

        self.task_type = task_type
        self.average = average

    def __call__(self, logits, target):
        if self.task_type == 'binary':
            self.y_prob = logits.sigmoid().data.cpu().numpy()
        else:
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def value(self):
        auc = roc_auc_score(y_score=self.y_prob, y_true=self.y_true, average=self.average,
                            multi_class='ovr')
        return auc

    def name(self):
        return 'auc'


class F1Score(Metric):
    """
    F1 Score
    binary:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = F1Score(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    """

    def __init__(self, thresh=0.5, normalizate=True, task_type='binary', average='binary', search_thresh=False):
        super(F1Score).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self, y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def __call__(self, logits, target):
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh).astype(int)
                self.value()
            else:
                thresh, f1 = self.thresh_search(y_prob=y_prob)
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob, 1)

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

    def name(self):
        return 'f1'


class ClassReport(Metric):
    """
    class report
    """

    def __init__(self, target_names=None):
        super(ClassReport).__init__()
        self.target_names = target_names

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        score = classification_report(y_true=self.y_true,
                                      y_pred=self.y_pred,
                                      target_names=self.target_names)
        print(f"\n\n classification report: {score}")

    def __call__(self, logits, target):
        """
        @param logits: tensor with size [N, num_class]
        @param target: tensor with size [N]
        @return:
        """
        _, y_pred = logits.argmax(dim=1)
        self.y_pred = y_pred.cpu().numpy()
        self.y_true = target.cpu().numpy()

    def name(self):
        return "class_report"


class MultiLabelReport(Metric):
    '''
    multi label report
    '''

    def __init__(self, id2label=None):
        super(MultiLabelReport).__init__()
        self.id2label = id2label
        self.y_prob = 0
        self.y_true = 0

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def __call__(self, logits, target):
        self.y_prob = logits.sigmoid().data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def value(self):
        for i, label in self.id2label.items():
            auc = roc_auc_score(y_score=self.y_prob[:, i], y_true=self.y_true[:, i])
            print(f"label:{label} - auc: {auc:.4f}")

    def name(self):
        return "multilabel_report"


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def calculate_rouge(
        pred_lns: List[str],
        tgt_lns: List[str],
        multi_refs=False,
        use_stemmer=True,
        rouge_keys=ROUGE_KEYS,
        return_precision_and_recall=False,
        bootstrap_aggregation=True,
        newline_sep=False,
) -> Dict:
    """Calculate rouge using rouge_scorer package.
    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).
    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys
    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(pred_lns, tgt_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if not multi_refs:
            scores = scorer.score(prediction=pred, target=tgt)
        else:
            scores = {}
            for tgt_i in tgt:
                score_i = scorer.score(prediction=pred, target=tgt_i)
                for key, values in score_i.items():
                    if key in scores:
                        scores[key] = max(score_i[key], scores[key])
                    else:
                        scores[key] = score_i[key]
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


class Rouge(Metric):
    """
    can: list(str)
    ref: list(str)
    """

    def __init__(self):
        super(Rouge).__init__()
        self.rouge = None

    def __call__(self, pred_str, label_str, multi_refs=False):
        self.rouge: Dict = calculate_rouge(pred_lns=pred_str, tgt_lns=label_str, multi_refs=multi_refs)

    def reset(self):
        self.rouge = None

    def value(self):
        return self.rouge

    def name(self):
        return "Rouge"


def calculate_bleu(output_lns, refs_lns, ngram=2, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation.
    output_lns = str
    ref_lns = list(str)
    ngram： int default: 2
    sentence_bleu params:
    references: [['this', 'is', 'a', 'test], ['this', 'is', 'test', 'too']]
    hypothesis: ['a', 'test', 'here'],
    weights=(0.25, 0.25, 0.25, 0.25),"""
    weights = [0.0, 0.0, 0.0, 0.0]
    for i in range(ngram):
        weights[i] = round(1.0 / ngram, 2)
    return {"bleu": round(sentence_bleu(references=[r.split() for r in refs_lns],
                                        hypothesis=output_lns.split(),
                                        weights=weights), 4)}


class BLEU(Metric):
    """
        can: str
        ref: list(str)
        """

    def __init__(self, n_gram=2):
        super(BLEU).__init__()
        self.bleu = None
        self.n_gram = n_gram

    def __call__(self, pred_str, label_str):
        self.bleu: Dict = calculate_bleu(output_lns=pred_str, refs_lns=label_str, ngram=self.n_gram)

    def reset(self):
        self.bleu = None

    def value(self):
        return self.bleu

    def name(self):
        return f"BLEU-{self.n_gram}"

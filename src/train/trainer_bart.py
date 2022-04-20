import torch
import pandas as pd

from ..callback.progressbar import ProgressBar
from ..common.tools import model_device
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from .losses import label_smoothed_nll_loss, cross_entropy_loss
from torch.nn.utils import clip_grad_norm_
from configs.basic_config import config


class Trainer(object):
    def __init__(self, args, model, tokenizer, logger, optimizer, scheduler, early_stopping,
                 epoch_metrics, batch_metrics, training_monitor=None, model_checkpoint=None,
                 verbose=1, position_weight=None, loss_ignore_index=None, input_len=None,
                 softmax_eps=1e-6, smooth_loss_epsilon=0.1):
        self.args = args
        self.model = model
        self.tok = tokenizer
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.training_monitor = training_monitor
        self.model_checkpoint = model_checkpoint

        self.verbose = verbose
        self.position_weight = position_weight
        self.loss_ignore_index = loss_ignore_index
        self.input_len = input_len

        self.softmax_eps = softmax_eps
        self.smooth_loss_epsilon = smooth_loss_epsilon

        self.start_epoch = 1
        self.global_step = 0
        self.model, self.device = model_device(n_gpu=args.n_gpu, model=self.model)

        if args.resume_path:
            resume_path = f"{args.resume_path}/epoch_{args.resume_epoch}_bart-base_model.bin"
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(f"{resume_path}")
            best = resume_dict['best']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

        self.outputs_ids = []
        self.targets = []
        self.pred_strs = []
        self.label_strs = []
        self.result = {}
        self.info = {}
        self.epoch_loss_agg = AverageMeter()

    def epoch_reset(self):
        self.outputs_ids = []
        self.targets = []
        self.result = {}
        self.pred_strs = []
        self.label_strs = []
        self.epoch_loss_agg = AverageMeter()
        if self.epoch_metrics is not None:
            for metric in self.epoch_metrics:
                metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save.state_dict(),
                 'epoch': epoch,
                 'best': best}
        return state

    def run_batch(self, batch, step, mode):
        self.batch_reset()
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, labels_masks, cls_label = batch

        if self.input_len is not None:
            input_ids = input_ids[:, :self.input_len]
            input_mask = input_mask[:, :self.input_len]
            segment_ids = segment_ids[:, :self.input_len]

        decoder_input_ids = label_ids[:, :-1]
        decoder_mask = labels_masks[:, :-1]
        target_ids = label_ids[:, 1:]

        outputs = self.model(input_ids=input_ids,
                             attention_mask=input_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_mask,
                             return_dict=True)

        # print(f"type of model outputs: {type(outputs)}")  # <class 'transformers.modeling_outputs.Seq2SeqLMOutput'>
        if "dict" in str(type(outputs)):
            logits = outputs["lm_logits"]
        else:
            logits = outputs.logits
        # [batch_size, ln, vocab_size]
        cls_logits = None

        logits_softmax = logits.softmax(dim=-1)
        logits_softmax = torch.where(logits_softmax < self.softmax_eps,
                                     torch.tensor(self.softmax_eps, dtype=torch.float32).to(self.device),
                                     logits_softmax)
        logit_log_softmax = logits_softmax.log()

        loss, nll_loss = label_smoothed_nll_loss(lprobs=logit_log_softmax,
                                                 target=target_ids,
                                                 epsilon=self.smooth_loss_epsilon,
                                                 ignore_index=self.loss_ignore_index,
                                                 position_weight=self.position_weight)
        loss = loss.mean()

        if mode == "train":
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
        else:  # mode == valid
            loss = loss

        self.info[f'loss'] = loss.item()
        self.epoch_loss_agg.update(loss.item(), n=1)

        # assert next_tokens has shape [batch_size, length]
        output_ids = torch.argmax(logits_softmax, dim=-1)

        return {"output_ids": output_ids,
                "logits": logits,
                "target_ids": target_ids,
                "cls_logits": cls_logits,
                "cls_label": cls_label}

    def model_fit(self, data, mode, print_interval, pbar):
        for step, batch in enumerate(data):
            batch_outputs = self.run_batch(batch, step=step, mode=mode)

            output_ids = batch_outputs["output_ids"]
            logits = batch_outputs["logits"]
            target_ids = batch_outputs["target_ids"]

            batch_pred_strs = self.tok.batch_decode(output_ids, skip_special_tokens=True)
            batch_target_strs = self.tok.batch_decode(target_ids, skip_special_tokens=True)

            if self.batch_metrics:
                for metric in self.batch_metrics:
                    if metric.name() == "Rouge":
                        metric(pred_str=batch_pred_strs, label_str=batch_target_strs)
                        self.info[metric.name()] = metric.value()
                    else:
                        metric(logits=logits.contiguous().view(-1, logits.shape[-1]),
                               target=target_ids.contiguous().view(-1))
                        self.info[metric.name()] = metric.value()

            if step % print_interval == 0:
                if self.verbose >= 1:
                    show_info = pbar(step=step, info=self.info)
                    self.logger.info(show_info)

            self.outputs_ids.append(output_ids.cpu().detach())
            self.targets.append(target_ids.cpu().detach())
            self.pred_strs += batch_pred_strs  # list of strings
            self.label_strs += batch_target_strs

    def run_epoch(self, data, mode, filename=None):
        length_data = len(data)
        print_interval = max(length_data // 10, 1)
        self.epoch_reset()
        if mode == "train":
            self.model.train()
            pbar = ProgressBar(n_total=length_data, desc='Training')
            self.model_fit(data, mode=mode, print_interval=print_interval, pbar=pbar)
            self.logger.info("\n------------- train result --------------")
            self.result['loss'] = self.epoch_loss_agg.avg

            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    if metric.name() == "Rouge":
                        metric(pred_str=self.pred_strs, label_str=self.label_strs)
                    else:
                        metric(logits=self.outputs_ids, target=self.targets)
                    value = metric.value()
                    if value:
                        if isinstance(value, dict):
                            for k, v in value.items():
                                self.result[f"{k}"] = v
                        else:
                            self.result[f'{metric.name()}'] = value
        else:
            self.model.eval()
            pbar = ProgressBar(n_total=length_data, desc='Evaluating')
            with torch.no_grad():
                self.model_fit(data, mode=mode, print_interval=print_interval, pbar=pbar)

                print("------------- valid result --------------")
                self.result['valid_loss'] = self.epoch_loss_agg.avg
                if self.epoch_metrics:
                    for metric in self.epoch_metrics:
                        if metric.name() == "Rouge":
                            metric(self.pred_strs, self.label_strs)
                        else:
                            metric(logits=self.outputs_ids, target=self.targets)
                        value = metric.value()
                        if value:
                            if isinstance(value, dict):
                                for k, v in value.items():
                                    self.result[f"valid_{k}"] = v
                            else:
                                self.result[f'valid_{metric.name()}'] = value

            if filename is not None:
                self.logger.info(f"Save valid cases to {filename}!")
                self.save_case(self.pred_strs, self.label_strs, filename)

        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self, train_data, valid_data):
        # ***************************************************************
        self.model.zero_grad()
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 a
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            train_log = self.run_epoch(train_data, mode="train")
            valid_result_dir = f'{config["result_dir"]}/{self.args.model_name}'
            import os
            if os.path.exists(valid_result_dir):
                pass
            else:
                os.mkdir(valid_result_dir)

            valid_log = self.run_epoch(valid_data,
                                       mode="valid",
                                       filename=valid_result_dir + f"/{self.args.arch}_{self.args.data_name}_{epoch}")
            logs = dict(train_log, **valid_log)

            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            self.logger.info(show_info)

            # save
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                self.logger.info(f"save model_checkpoints")
                state = self.save_info(epoch, best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.epoch_step(current=logs[self.model_checkpoint.monitor], state=state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break

    def save_case(self, pred, label, file):
        assert len(pred) == len(label)
        df_gen = pd.DataFrame(columns=['generated', 'reference'])
        df_gen['generated'] = pred
        df_gen['reference'] = label
        df_gen.to_csv(f"{file}.csv")


class TrainerCls(object):
    def __init__(self, args, model, tokenizer, logger, optimizer, scheduler, early_stopping,
                 epoch_metrics, batch_metrics, training_monitor=None, model_checkpoint=None,
                 verbose=1, position_weight=None, loss_ignore_index=None, input_len=None,
                 softmax_eps=1e-6, smooth_loss_epsilon=0.1):
        self.args = args
        self.model = model
        self.tok = tokenizer
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.training_monitor = training_monitor
        self.model_checkpoint = model_checkpoint

        self.verbose = verbose
        self.position_weight = position_weight
        self.loss_ignore_index = loss_ignore_index
        self.input_len = input_len

        self.softmax_eps = softmax_eps
        self.smooth_loss_epsilon = smooth_loss_epsilon

        self.start_epoch = 1
        self.global_step = 0
        self.model, self.device = model_device(n_gpu=args.n_gpu, model=self.model)

        if args.resume_path:
            resume_path = f"{args.resume_path}/epoch_{args.resume_epoch}_bart-base_model.bin"
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(f"{resume_path}")
            best = resume_dict['best']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

        self.outputs_ids = []
        self.targets = []
        self.pred_strs = []
        self.label_strs = []
        self.cls_logits = []
        self.cls_labels = []

        self.result = {}
        self.info = {}
        self.epoch_loss_agg = AverageMeter()
        self.epoch_clsloss_agg = AverageMeter()

    def epoch_reset(self):
        self.outputs_ids = []
        self.targets = []
        self.pred_strs = []
        self.label_strs = []
        self.cls_logits = []
        self.cls_labels = []

        self.result = {}
        self.epoch_loss_agg = AverageMeter()
        self.epoch_clsloss_agg = AverageMeter()
        if self.epoch_metrics is not None:
            for metric in self.epoch_metrics:
                metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save.state_dict(),
                 'epoch': epoch,
                 'best': best}
        return state

    def run_batch(self, batch, step, mode, eps=1e-6, epsilon=0.1):
        self.batch_reset()
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, labels_masks, cls_label = batch

        if self.input_len is not None:
            input_ids = input_ids[:, :self.input_len]
            input_mask = input_mask[:, :self.input_len]
            segment_ids = segment_ids[:, :self.input_len]

        decoder_input_ids = label_ids[:, :-1]
        decoder_mask = labels_masks[:, :-1]
        target_ids = label_ids[:, 1:]
        target_mask = labels_masks[:, 1:]

        outputs = self.model(input_ids=input_ids,
                             attention_mask=input_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_mask,
                             labels=target_ids.contiguous(),
                             return_dict=True)

        # [batch_size, ln, vocab_size]
        if isinstance(outputs, dict):
            logits = outputs["lm_logits"]
        else:
            logits = outputs.logits
        cls_logits = outputs["cls_logits"]

        label_indice = cls_label.eq(0).nonzero(as_tuple=True)[0]  # [bsz]
        # select indice of which the label is 0

        if label_indice.size(0) != 0:
            # select logits has label 0
            logits = logits.index_select(dim=0, index=label_indice)
            target_ids = target_ids.index_select(dim=0, index=label_indice)

            logits_softmax = logits.softmax(dim=-1)
            logits_softmax = torch.where(logits_softmax < eps,
                                         torch.tensor(eps, dtype=torch.float32).to(self.device),
                                         logits_softmax)
            logit_log_softmax = logits_softmax.log()
            # assert next_tokens has shape [batch_size, length]
            output_ids = torch.argmax(logits_softmax, dim=-1)

            loss, nll_loss = label_smoothed_nll_loss(lprobs=logit_log_softmax,
                                                     target=target_ids,
                                                     epsilon=epsilon,
                                                     ignore_index=self.loss_ignore_index,
                                                     position_weight=self.position_weight)
            # print(f"lm loss shape: {loss.size()}")
            loss = loss.mean()
            self.info[f'loss'] = loss.item()
            self.epoch_loss_agg.update(loss.item(), n=label_indice.size(0))
        else:
            logits = torch.tensor([])
            output_ids = torch.tensor([])
            target_ids = torch.tensor([])
            loss = 0.0
            self.info[f'loss'] = loss

        cls_loss = cross_entropy_loss(cls_logits, cls_label)
        self.info['clsloss'] = cls_loss.item()
        self.epoch_clsloss_agg.update(cls_loss.item(), n=cls_logits.size(0))

        loss += cls_loss

        if mode == "train" and loss > 0:
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

        return {"output_ids": output_ids,
                "logits": logits,
                "target_ids": target_ids,
                "cls_logits": cls_logits,
                "cls_label": cls_label,
                "target_mask": target_mask}

    def model_fit(self, data, mode, print_interval, pbar):
        for step, batch in enumerate(data):
            batch_outputs = self.run_batch(batch, step=step, mode=mode)

            output_ids = batch_outputs["output_ids"]
            logits = batch_outputs["logits"]
            target_ids = batch_outputs["target_ids"]
            cls_logits = batch_outputs["cls_logits"]
            cls_label = batch_outputs["cls_label"]

            if output_ids.size(0):
                batch_pred_strs = self.tok.batch_decode(output_ids, skip_special_tokens=True)
                batch_target_strs = self.tok.batch_decode(target_ids, skip_special_tokens=True)

                if self.batch_metrics:
                    for metric in self.batch_metrics:
                        if metric.name() == "Rouge":
                            metric(pred_str=batch_pred_strs, label_str=batch_target_strs)
                            self.info[metric.name()] = metric.value()
                        elif metric.name() == "accuracy":
                            metric(logits=logits.contiguous().view(-1, logits.shape[-1]),
                                   target=target_ids.contiguous().view(-1))
                            self.info[metric.name()] = metric.value()
                        else:
                            metric(logits=logits.contiguous().view(-1, logits.shape[-1]),
                                   target=target_ids.contiguous().view(-1))
                            self.info[metric.name()] = metric.value()
            else:
                batch_pred_strs = []
                batch_target_strs = []
                for metric in self.batch_metrics:
                    self.info[metric.name()] = -1.0

            from src.train.metrics import Accuracy
            cls_accuracy = Accuracy(topK=1)
            cls_accuracy(logits=cls_logits.contiguous().view(-1, cls_logits.shape[-1]),
                         target=cls_label.contiguous().view(-1))
            value = cls_accuracy.value()
            self.info[f"cls-{cls_accuracy.name()}"] = value

            if step % print_interval == 0:
                if self.verbose >= 1:
                    show_info = pbar(step=step, info=self.info)
                    self.logger.info(show_info)

            self.outputs_ids.append(output_ids.cpu().detach())
            self.targets.append(target_ids.cpu().detach())
            self.pred_strs += batch_pred_strs  # list of strings
            self.label_strs += batch_target_strs
            self.cls_logits.append(cls_logits)
            self.cls_labels.append(cls_label)

    def run_epoch(self, data, mode, filename=None):
        length_data = len(data)
        print_interval = max(length_data // 10, 1)
        self.epoch_reset()
        if mode == "train":
            self.model.train()
            pbar = ProgressBar(n_total=length_data, desc='Training')
            self.model_fit(data, mode, print_interval, pbar)
            epoch_cls_logits = torch.cat(self.cls_logits, dim=0)
            epoch_cls_labels = torch.cat(self.cls_labels, dim=0)
            self.logger.info("\n------------- train result --------------")
            # epoch metric
            # the epoch of logits is out-of-memory, so we cannot use logits in epoch calculation,
            # only maintain the ids and strings for epoch metric!!!!!
            # for rouge, we only calculate the rouge metric when validation
            self.result['loss'] = self.epoch_loss_agg.avg
            self.result['clsloss'] = self.epoch_clsloss_agg.avg

            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    if metric.name() == "Rouge":
                        metric(pred_str=self.pred_strs, label_str=self.label_strs)
                    else:
                        metric(logits=epoch_cls_logits,
                               target=epoch_cls_labels)
                    value = metric.value()
                    if value:
                        if isinstance(value, dict):
                            for k, v in value.items():
                                self.result[f"{k}"] = v
                        else:
                            self.result[f'{metric.name()}'] = value
        else:
            self.model.eval()
            pbar = ProgressBar(n_total=length_data, desc='Evaluating')
            with torch.no_grad():
                self.model_fit(data, mode, print_interval, pbar)
            epoch_cls_logits = torch.cat(self.cls_logits, dim=0)
            epoch_cls_labels = torch.cat(self.cls_labels, dim=0)
            print("------------- valid result --------------")
            self.result['valid_loss'] = self.epoch_loss_agg.avg
            self.result['valid_clsloss'] = self.epoch_clsloss_agg.avg
            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    if metric.name() == "Rouge":
                        metric(self.pred_strs, self.label_strs)
                    else:
                        metric(logits=epoch_cls_logits,
                               target=epoch_cls_labels)
                    value = metric.value()
                    if value:
                        if isinstance(value, dict):
                            for k, v in value.items():
                                self.result[f"valid_{k}"] = v
                        else:
                            self.result[f'valid_{metric.name()}'] = value

            if filename is not None:
                self.logger.info(f"Save valid cases to {filename}!")
                self.save_case(self.pred_strs, self.label_strs, filename)

        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self, train_data, valid_data):
        # ***************************************************************
        self.model.zero_grad()
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 a
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            train_log = self.run_epoch(train_data, mode="train")
            valid_result_dir = f'{config["result_dir"]}/{self.args.model_name}'
            import os
            if os.path.exists(valid_result_dir):
                pass
            else:
                os.mkdir(valid_result_dir)

            valid_log = self.run_epoch(
                valid_data,
                mode="valid",
                filename=valid_result_dir + f"/{self.args.arch}_{self.args.data_name}_{epoch}")
            logs = dict(train_log, **valid_log)

            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            self.logger.info(show_info)

            # save
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                self.logger.info(f"save model_checkpoints")
                state = self.save_info(epoch, best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.epoch_step(current=logs[self.model_checkpoint.monitor], state=state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break

    def save_case(self, pred, label, file):
        assert len(pred) == len(label)
        df_gen = pd.DataFrame(columns=['generated', 'reference'])
        df_gen['generated'] = pred
        df_gen['reference'] = label
        df_gen.to_csv(f"{file}.csv")

import torch
import torch.nn.functional as F
import pandas as pd
from torch.nn.utils import clip_grad_norm_

from ..callback.progressbar import ProgressBar
from ..common.tools import model_device
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from .losses import label_smoothed_nll_loss, cross_entropy_loss, kld_loss
from configs.basic_config import config


class Trainer(object):
    def __init__(self, args, model, kg_model, tok, logger, optimizer, scheduler, early_stopping,
                 epoch_metrics, batch_metrics, training_monitor=None, model_checkpoint=None,
                 verbose=1, position_weight=None, loss_ignore_index=None, input_len=None,
                 softmax_eps=1e-6, smooth_loss_epsilon=0.1):
        self.args = args
        self.model = model
        self.kg_model = kg_model

        self.tok = tok
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
            resume_path = config["checkpoint_dir"] / f"{args.resume_path}/epoch_{args.resume_epoch}_bart-base_model.bin"
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
        self.epoch_clsloss_agg = AverageMeter()

    def epoch_reset(self):
        self.outputs_ids = []
        self.targets = []
        self.pred_strs = []
        self.label_strs = []

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

    def create_im_inputs(self, input_ids, input_mask, segment_ids, event_only=False):
        batch_size, ln = input_ids.size()

        if event_only:
            event_ids = input_ids.masked_select(segment_ids)
            event_lens = [event_ids[i, :].size(-1) for i in range(batch_size)]
        else:
            event_ids = input_ids
            event_lens = input_mask.sum(dim=-1)

        char_relation = self.tok(config['relations'], padding=True, return_length=True, return_tensors='pt')
        relation_ids = char_relation["input_ids"].to(input_ids.device)
        relation_mask = char_relation["attention_mask"].to(input_ids.device)
        relation_lens = relation_mask.sum(dim=-1)

        dim = relation_ids.size(0)

        # event_ids = input_ids.masked_select(segment_ids.eq(1)).view(size=[input_ids.size(0), -1])
        # event_attention_mask = input_mask.masked_select(segment_ids.eq(1)).view(size=[input_ids.size(0), -1])

        kg_input_ids = []
        for i in range(batch_size):
            cur_kg_input_ids = []
            cur_input_ids = event_ids[i]
            cur_ln = event_lens[i]
            for j in range(dim):
                char_ln = relation_lens[j]
                cur_relation_ids = relation_ids[j]
                if cur_ln + char_ln > ln:
                    it = torch.cat([cur_input_ids[:ln - char_ln - 1],
                                    cur_input_ids[cur_ln - 1:cur_ln],
                                    cur_relation_ids[:char_ln]],
                                   dim=-1).unsqueeze(0)
                    cur_kg_input_ids.append(it)
                else:
                    it = torch.cat(
                        [cur_input_ids[:cur_ln],
                         cur_relation_ids[:char_ln],
                         cur_input_ids[cur_ln + char_ln:]],
                        dim=-1).unsqueeze(0)
                    cur_kg_input_ids.append(it)
            kg_input_ids.append(torch.cat(cur_kg_input_ids, dim=0).unsqueeze(0))
        kg_input_ids = torch.cat(kg_input_ids, dim=0)  # (batch_size, dim, -1)
        kg_input_masks = kg_input_ids.ne(self.tok.pad_token_id).int()

        kg_input_ids = kg_input_ids.reshape([batch_size * dim, -1])
        kg_input_masks = kg_input_masks.reshape([batch_size * dim, -1])
        return kg_input_ids, kg_input_masks

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

        batch_size, ln = input_ids.size()
        kg_input_ids, kg_input_masks = self.create_im_inputs(input_ids, input_mask, segment_ids, event_only=True)

        kg_batch_size, kg_len = kg_input_ids.size()

        outputs = self.model(input_ids=input_ids,
                             attention_mask=input_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_mask,
                             kg_input_ids=kg_input_ids,
                             kg_input_masks=kg_input_masks,
                             return_dict=True)

        # print(f"type of model outputs: {type(outputs)}")  # <class 'transformers.modeling_outputs.Seq2SeqLMOutput'>
        if "dict" in str(type(outputs)):
            logits = outputs["lm_logits"]
        else:
            logits = outputs.logits
        # [batch_size, ln, vocab_size]
        explanation_output_ids = None

        logits_softmax = logits.softmax(dim=-1)
        logits_softmax = torch.where(logits_softmax < self.softmax_eps,
                                     torch.tensor(self.softmax_eps, dtype=torch.float32).to(self.device),
                                     logits_softmax)
        logit_log_softmax = logits_softmax.log()
        # assert next_tokens has shape [batch_size, length]
        output_ids = torch.argmax(logits_softmax, dim=-1)

        loss, nll_loss = label_smoothed_nll_loss(lprobs=logit_log_softmax,
                                                 target=target_ids,
                                                 epsilon=self.smooth_loss_epsilon,
                                                 ignore_index=self.loss_ignore_index,
                                                 position_weight=self.position_weight)
        # print(f"lm loss shape: {loss.size()}")
        loss = loss.mean().view(1)
        explanation_kld_loss = torch.tensor([0], dtype=torch.float32).to(self.device)
        explanation_cls_loss = torch.tensor([0], dtype=torch.float32).to(self.device)

        if self.args.training_explanation:
            # use fixed kg_model decoder to generate the explanations
            kg_model = self.kg_model.to(self.device)
            kg_model.requires_grad_(False)

            kg_hidden_states = outputs["kg_hidden_states"]  # [bsz, dim, ln, E]

            kg_decoder_input_ids = torch.full([kg_batch_size, 1], fill_value=self.tok.bos_token_id, dtype=torch.long).to(
                self.device)
            kg_decoder_input_mask = torch.ones_like(kg_decoder_input_ids, dtype=torch.long).to(self.device)

            kg_encoder_hidden_states = kg_hidden_states.reshape([batch_size * dim, ln, -1]).to(self.device)
            kg_encoder_padding_mask = kg_input_masks.reshape([batch_size * dim, ln]).to(self.device)
            kg_input_ids = kg_input_ids.reshape([batch_size * dim, -1]).to(self.device)

            explanation_len = self.args.explanation_len if self.args.explanation_len is not None else self.args.max_decode_len

            kg_decoder = kg_model.model.get_decoder()
            kg_embeddings_weight = kg_model.model.get_input_embeddings()
            kg_decoder_word_emb = kg_embeddings_weight(kg_decoder_input_ids)

            finished_sentences = torch.tensor([False] * (batch_size * dim)).to(self.device)

            for i in range(explanation_len):
                decoder_outputs = kg_decoder(inputs_embeds=kg_decoder_word_emb,
                                             attention_mask=kg_decoder_input_mask,
                                             encoder_hidden_states=kg_encoder_hidden_states,
                                             encoder_attention_mask=kg_encoder_padding_mask,
                                             return_dict=True)

                vocab_logits = kg_model.lm_head(decoder_outputs[0]) + kg_model.final_logits_bias

                vocab_logits = vocab_logits[:, -1, :]

                if self.args.use_kld_loss:
                    lm_outputs = kg_model(input_ids=kg_input_ids,
                                          attention_mask=kg_encoder_padding_mask,
                                          decoder_inputs_embeds=kg_decoder_word_emb,
                                          decoder_attention_mask=kg_decoder_input_mask, )

                    lm_vocab_logits = lm_outputs["lm_logits"][:, -1, :]

                    lm_vocab_probs = torch.softmax(lm_vocab_logits, dim=-1)
                    vocab_logprobs = torch.log_softmax(vocab_logits, dim=-1)
                    explanation_kld_loss += kld_loss(vocab_logprobs, lm_vocab_probs)

                oh_vocab_probs = F.gumbel_softmax(vocab_logits, hard=False).unsqueeze(1)  # output soft probs
                chosen_emb = torch.matmul(oh_vocab_probs, kg_embeddings_weight.weight)  # [B, E]

                max_prob_idx = torch.argmax(oh_vocab_probs, dim=-1).squeeze()
                # [B] predicted next word according to the gumbel-softmax probs

                next_word = max_prob_idx.masked_fill(finished_sentences.eq(True),
                                                     self.tok.pad_token_id)
                finished_sentences = finished_sentences.masked_fill(max_prob_idx.eq(self.tok.eos_token_id), True)

                kg_decoder_input_ids = torch.cat([kg_decoder_input_ids, next_word.unsqueeze(1)], dim=-1)
                kg_decoder_word_emb = torch.cat((kg_decoder_word_emb, chosen_emb), dim=1)
                kg_decoder_input_mask = kg_decoder_input_ids.ne(self.tok.pad_token_id).int().to(self.device)

            explanation_kld_loss = explanation_kld_loss / explanation_len

            if self.args.use_cls_loss:
                kg_decoder_hidden_states = decoder_outputs.last_hidden_state
                eos_mask = kg_decoder_input_ids[:, :-1].eq(self.tok.eos_token_id)
                eos_mask[eos_mask.sum(1).eq(0), -1] = True
                if len(torch.unique(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")
                eos_hidden_states = kg_decoder_hidden_states[eos_mask, :].view(
                    kg_decoder_hidden_states.size(0), -1, kg_decoder_hidden_states.size(-1))[:, -1, :]  # [bsz, E]
                lm_cls_logits = kg_model.classification_head(eos_hidden_states)  # [bsz, num_labels]
                explanation_cls_label = torch.zeros([lm_cls_logits.size(0)], dtype=torch.long).to(
                    self.device)
                explanation_cls_loss += cross_entropy_loss(lm_cls_logits, explanation_cls_label)

        self.info[f'loss'] = loss.item()
        if self.args.training_explanation:
            if self.args.use_kld_loss:
                self.info[f"explanation-kld-loss"] = explanation_kld_loss.item()
            if self.args.use_cls_loss:
                self.info[f"explanation-cls-loss"] = explanation_cls_loss.item()

        self.epoch_loss_agg.update(
            self.args.explanation_coeff * loss.item() + (explanation_kld_loss.item() + explanation_cls_loss.item()),
            n=batch_size)

        # print(f"loss: {loss.size()}, "
        #       f"explanation_kld_loss: {explanation_kld_loss.size()},"
        #       f"explanation_cls_loss: {explanation_cls_loss.size()}")
        loss = self.args.explanation_coeff * loss + (explanation_kld_loss + explanation_cls_loss)

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
                "cls_logits": None,
                "cls_label": cls_label,
                "explanation_output_ids": explanation_output_ids}

    def model_fit(self, data, mode, print_interval, pbar):
        for step, batch in enumerate(data):
            batch_outputs = self.run_batch(batch, step=step, mode=mode)

            output_ids = batch_outputs["output_ids"]
            logits = batch_outputs["logits"]
            target_ids = batch_outputs["target_ids"]
            explanation_output_ids = batch_outputs["explanation_output_ids"]

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

            if step % print_interval == 0:
                if self.verbose >= 1:
                    show_info = pbar(step=step, info=self.info)
                    self.logger.info(show_info)
                    # if explanation_output_ids is not None:
                    #     self.logger.info(
                    #         f"example of explanation: {self.tok.batch_decode(explanation_output_ids[:9, :])}")

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
            self.model_fit(data, mode, print_interval, pbar)
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
                        metric(self.pred_strs, self.label_strs)
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
            print("------------- valid result --------------")
            self.result['valid_loss'] = self.epoch_loss_agg.avg
            self.result['valid_clsloss'] = self.epoch_clsloss_agg.avg
            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    if metric.name() == "Rouge":
                        metric(self.pred_strs, self.label_strs)
                    else:
                        metric(self.pred_strs, self.label_strs)
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


class TrainerCls(object):
    def __init__(self, args, model, kg_model, tok, logger, optimizer, scheduler, early_stopping,
                 epoch_metrics, batch_metrics, training_monitor=None, model_checkpoint=None,
                 verbose=1, position_weight=None, loss_ignore_index=None, input_len=None,
                 softmax_eps=1e-6, smooth_loss_epsilon=0.1):
        self.args = args
        self.model = model
        self.kg_model = kg_model

        self.tok = tok
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
            resume_path = config["checkpoint_dir"] / f"{args.resume_path}/epoch_{args.resume_epoch}_bart-base_model.bin"
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

        batch_size, ln = input_ids.size()

        char_relation = self.tok(config['relations'], padding=True, return_length=True, return_tensors='pt')
        relation_ids = char_relation["input_ids"].to(input_ids.device)
        relation_mask = char_relation["attention_mask"].to(input_ids.device)

        dim = relation_ids.size(0)

        # event_ids = input_ids.masked_select(segment_ids.eq(1)).view(size=[input_ids.size(0), -1])
        # event_attention_mask = input_mask.masked_select(segment_ids.eq(1)).view(size=[input_ids.size(0), -1])

        event_ids = input_ids
        kg_input_ids = []
        for i in range(batch_size):
            cur_kg_input_ids = []
            cur_input_ids = event_ids[i]
            cur_ln = input_mask[i].sum()
            for j in range(dim):
                char_ln = relation_mask[j].sum()
                cur_relation_ids = relation_ids[j]
                if cur_ln + char_ln > ln:
                    it = torch.cat([cur_input_ids[:ln - char_ln - 1], cur_input_ids[cur_ln - 1:cur_ln],
                                    cur_relation_ids[:char_ln]],
                                   dim=-1).unsqueeze(0)
                    cur_kg_input_ids.append(it)
                else:
                    it = torch.cat(
                        [cur_input_ids[:cur_ln], cur_relation_ids[:char_ln], cur_input_ids[cur_ln + char_ln:]],
                        dim=-1).unsqueeze(0)
                    cur_kg_input_ids.append(it)
            kg_input_ids.append(torch.cat(cur_kg_input_ids, dim=0).unsqueeze(0))
        kg_input_ids = torch.cat(kg_input_ids, dim=0)
        kg_input_masks = kg_input_ids.ne(self.tok.pad_token_id).int()

        outputs = self.model(input_ids=input_ids,
                             attention_mask=input_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_mask,
                             kg_input_ids=kg_input_ids,
                             kg_input_masks=kg_input_masks,
                             return_dict=True)

        # print(f"type of model outputs: {type(outputs)}")  # <class 'transformers.modeling_outputs.Seq2SeqLMOutput'>
        if "dict" in str(type(outputs)):
            logits = outputs["lm_logits"]
        else:
            logits = outputs.logits
        # [batch_size, ln, vocab_size]

        cls_logits = outputs["cls_logits"]

        explanation_output_ids = None

        label_indice = cls_label.eq(0).nonzero(as_tuple=True)[0]  # [bsz]

        if label_indice.size(0) != 0:
            # select logits has label 0
            logits = logits.index_select(dim=0, index=label_indice)
            target_ids = target_ids.index_select(dim=0, index=label_indice)

            logits_softmax = logits.softmax(dim=-1)
            logits_softmax = torch.where(logits_softmax < self.softmax_eps,
                                         torch.tensor(self.softmax_eps, dtype=torch.float32).to(self.device),
                                         logits_softmax)
            logit_log_softmax = logits_softmax.log()
            # assert next_tokens has shape [batch_size, length]
            output_ids = torch.argmax(logits_softmax, dim=-1)

            loss, nll_loss = label_smoothed_nll_loss(lprobs=logit_log_softmax,
                                                     target=target_ids,
                                                     epsilon=self.smooth_loss_epsilon,
                                                     ignore_index=self.loss_ignore_index,
                                                     position_weight=self.position_weight)
            # print(f"lm loss shape: {loss.size()}")
            loss = loss.mean().view(1)
            explanation_kld_loss = torch.tensor([0], dtype=torch.float32).to(self.device)
            explanation_cls_loss = torch.tensor([0], dtype=torch.float32).to(self.device)

            if self.args.training_explanation:
                # use fixed kg_model decoder to generate the explanations
                kg_model = self.kg_model.to(self.device)
                kg_model.requires_grad_(False)

                kg_hidden_states = outputs["kg_hidden_states"]  # [bsz, dim, ln, E]
                kg_hidden_states = kg_hidden_states.index_select(dim=0, index=label_indice)
                kg_input_ids = kg_input_ids.index_select(dim=0, index=label_indice)
                kg_input_masks = kg_input_masks.index_select(dim=0, index=label_indice)

                true_data_bsz, dim, ln = kg_input_masks.size()

                kg_decoder_input_ids = torch.ones([true_data_bsz * dim, 1], dtype=torch.long).to(
                    self.device) * self.tok.bos_token_id
                kg_decoder_input_mask = torch.ones_like(kg_decoder_input_ids, dtype=torch.long).to(self.device)

                kg_encoder_hidden_states = kg_hidden_states.reshape([true_data_bsz * dim, ln, -1]).to(self.device)
                kg_encoder_padding_mask = kg_input_masks.reshape([true_data_bsz * dim, ln]).to(self.device)
                kg_input_ids = kg_input_ids.reshape([true_data_bsz * dim, -1]).to(self.device)

                explanation_len = self.args.explanation_len if self.args.explanation_len is not None else self.args.max_decode_len

                kg_decoder = kg_model.model.get_decoder()
                kg_embeddings_weight = kg_model.model.get_input_embeddings()
                kg_decoder_word_emb = kg_embeddings_weight(kg_decoder_input_ids)

                finished_sentences = torch.tensor([False] * (true_data_bsz * dim)).to(self.device)

                for i in range(explanation_len):
                    # print(f"kg_decoder_word_emb: {kg_decoder_word_emb.size()}, "
                    #       f"kg_decoder_input_mask: {kg_decoder_input_mask.size()}, "
                    #       f"kg_encoder_hidden_states: {kg_encoder_hidden_states.size()}, "
                    #       f"kg_encoder_padding_mask: {kg_encoder_padding_mask.size()}\n")
                    decoder_outputs = kg_decoder(inputs_embeds=kg_decoder_word_emb,
                                                 attention_mask=kg_decoder_input_mask,
                                                 encoder_hidden_states=kg_encoder_hidden_states,
                                                 encoder_attention_mask=kg_encoder_padding_mask,
                                                 return_dict=True)

                    vocab_logits = kg_model.lm_head(decoder_outputs[0]) + kg_model.final_logits_bias

                    vocab_logits = vocab_logits[:, -1, :]

                    if self.args.use_kld_loss:
                        lm_outputs = kg_model(input_ids=kg_input_ids,
                                              attention_mask=kg_encoder_padding_mask,
                                              decoder_inputs_embeds=kg_decoder_word_emb,
                                              decoder_attention_mask=kg_decoder_input_mask, )

                        lm_vocab_logits = lm_outputs["lm_logits"][:, -1, :]

                        lm_vocab_probs = torch.softmax(lm_vocab_logits, dim=-1)
                        vocab_logprobs = torch.log_softmax(vocab_logits, dim=-1)
                        explanation_kld_loss += kld_loss(vocab_logprobs, lm_vocab_probs)

                    oh_vocab_probs = F.gumbel_softmax(vocab_logits, hard=False).unsqueeze(1)  # output soft probs
                    chosen_emb = torch.matmul(oh_vocab_probs, kg_embeddings_weight.weight)  # [B, E]

                    max_prob_idx = torch.argmax(oh_vocab_probs, dim=-1).squeeze()
                    # [B] predicted next word according to the gumbel-softmax probs

                    next_word = max_prob_idx.masked_fill(finished_sentences.eq(True),
                                                         self.tok.pad_token_id)
                    finished_sentences = finished_sentences.masked_fill(max_prob_idx.eq(self.tok.eos_token_id), True)

                    kg_decoder_input_ids = torch.cat([kg_decoder_input_ids, next_word.unsqueeze(1)], dim=-1)
                    kg_decoder_word_emb = torch.cat((kg_decoder_word_emb, chosen_emb), dim=1)
                    kg_decoder_input_mask = kg_decoder_input_ids.ne(self.tok.pad_token_id).int().to(self.device)

                explanation_kld_loss = explanation_kld_loss / explanation_len

                if self.args.use_cls_loss:
                    kg_decoder_hidden_states = decoder_outputs.last_hidden_state
                    eos_mask = kg_decoder_input_ids[:, :-1].eq(self.tok.eos_token_id)
                    eos_mask[eos_mask.sum(1).eq(0), -1] = True
                    if len(torch.unique(eos_mask.sum(1))) > 1:
                        raise ValueError("All examples must have the same number of <eos> tokens.")
                    eos_hidden_states = kg_decoder_hidden_states[eos_mask, :].view(
                        kg_decoder_hidden_states.size(0), -1, kg_decoder_hidden_states.size(-1))[:, -1, :]  # [bsz, E]
                    lm_cls_logits = kg_model.classification_head(eos_hidden_states)  # [bsz, num_labels]
                    explanation_cls_label = torch.zeros([lm_cls_logits.size(0)], dtype=torch.long).to(
                        self.device)
                    explanation_cls_loss += cross_entropy_loss(lm_cls_logits, explanation_cls_label)

                explanation_output_ids = kg_decoder_input_ids[:, 1:]
                assert explanation_output_ids.size(0) == true_data_bsz * dim

            self.info[f'loss'] = loss.item()
            if self.args.training_explanation:
                if self.args.use_kld_loss:
                    self.info[f"explanation-kld-loss"] = explanation_kld_loss.item()
                if self.args.use_cls_loss:
                    self.info[f"explanation-cls-loss"] = explanation_cls_loss.item()

            self.epoch_loss_agg.update(
                loss.item() * self.args.explanation_coeff + (explanation_kld_loss.item() + explanation_cls_loss.item()),
                n=label_indice.size(0))

            # print(f"loss: {loss.size()}, "
            #       f"explanation_kld_loss: {explanation_kld_loss.size()},"
            #       f"explanation_cls_loss: {explanation_cls_loss.size()}")
            loss = self.args.explanation_coeff * loss + (explanation_kld_loss + explanation_cls_loss)
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
                "explanation_output_ids": explanation_output_ids}

    def model_fit(self, data, mode, print_interval, pbar):
        for step, batch in enumerate(data):
            batch_outputs = self.run_batch(batch, step=step, mode=mode)

            output_ids = batch_outputs["output_ids"]
            logits = batch_outputs["logits"]
            target_ids = batch_outputs["target_ids"]
            cls_logits = batch_outputs["cls_logits"]
            cls_label = batch_outputs["cls_label"]
            explanation_output_ids = batch_outputs["explanation_output_ids"]

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

            from .metrics import Accuracy
            cls_accuracy = Accuracy(topK=1)
            cls_accuracy(logits=cls_logits.contiguous().view(-1, cls_logits.shape[-1]),
                         target=cls_label.contiguous().view(-1))
            value = cls_accuracy.value()
            self.info[f"cls-{cls_accuracy.name()}"] = value

            if step % print_interval == 0:
                if self.verbose >= 1:
                    show_info = pbar(step=step, info=self.info)
                    self.logger.info(show_info)
                    # if explanation_output_ids is not None:
                    #     self.logger.info(
                    #         f"example of explanation: {self.tok.batch_decode(explanation_output_ids[:9, :])}")

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

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Callable, Tuple

from transformers.generation_logits_process import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation_beam_search import BeamSearchScorer


def init_sequence_length_for_generation(
        input_ids: torch.LongTensor, max_length: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    sequence_lengths = input_ids.new(input_ids.shape[0]).fill_(max_length)

    cur_len = input_ids.shape[-1]
    return sequence_lengths, unfinished_sequences, cur_len


def update_seq_length_for_generation(
        sequence_lengths: torch.LongTensor,
        unfinished_sequences: torch.LongTensor,
        cur_len: int,
        is_eos_in_next_token: torch.BoolTensor,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    # check if sentence is not finished yet
    is_sent_unfinished = unfinished_sequences.mul(is_eos_in_next_token.long()).bool()

    # update sentence length
    sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len)
    unfinished_sequences = unfinished_sequences.mul((~is_eos_in_next_token).long())
    return sequence_lengths, unfinished_sequences


def make_sampler(sampler_type, cfg, *args, **kwargs):
    print("Initializing Greedy Sampler")
    return GreedySampler(cfg, *args, **kwargs)


class Sampler():
    def __init__(self, cfg):
        # Token on which to end sampling
        self.cfg = cfg

    def generate_sequence(self, batch, model, start_idx, end_len):
        raise

    def get_logits_processor(self,
                             repetition_penalty: float = None,
                             no_repeat_ngram_size: int = None,
                             bad_words_ids: List[List[int]] = None,
                             min_length: int = None,
                             eos_token_id: int = None,
                             prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
                             num_beams: int = None,
                             ) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
        """

        # init warp parameters
        repetition_penalty = repetition_penalty if repetition_penalty is not None else 1.0
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else 0
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else None
        min_length = min_length if min_length is not None else 0
        eos_token_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        # instantiate processors list
        processors = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if bad_words_ids is not None:
            processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams))
        return processors

    def get_logits_warper(self, top_k: int = None, top_p: float = None, temperature: float = None, num_beams: int = None
                          ) -> LogitsProcessorList:
        """
                This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
                :obj:`~transformers.LogitsWarper` instances used for multinomial sampling.
                """

        # init warp parameters
        top_k = top_k if top_k is not None else 50
        top_p = top_p if top_p is not None else 1.0
        temperature = temperature if temperature is not None else 1.0  # all these default values are from bart_config
        # instantiate warpers list
        warpers = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        return warpers


class GreedySampler(Sampler):
    def __init__(self, cfg, batch_mode=True):
        super(GreedySampler, self).__init__(cfg)

    def append_batch(self, X, next_idx, mask):
        next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
        return torch.cat((X, next_idx), 1), next_mask

    def generate_sequence(self, batch, model, start_idx, end_len,
                          repetition_penalty=1.0,
                          no_repeat_ngram_size=0):
        logits_processor = self.get_logits_processor(repetition_penalty=repetition_penalty,
                                                     no_repeat_ngram_size=no_repeat_ngram_size)

        # print(f"batch: {batch}")
        input_ids = batch["input_ids"][:, :start_idx]
        input_attention_mask = batch["input_attention_mask"][:, :start_idx]
        encoder_hidden_states = batch["encoder_hidden_states"]
        encoder_padding_mask = batch["encoder_padding_mask"]

        history_states, history_attention = None, None

        if "history_states" in batch:
            history_states = batch["history_states"]
            history_attention = batch["history_attention_list"]

            outputs = model(input_ids=input_ids,
                            attention_mask=input_attention_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_padding_mask,
                            past_key_values=None,
                            inputs_embeds=None,
                            history_states_list=history_states,
                            history_attention_list=history_attention,
                            use_cache=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=True, )
        else:
            outputs = model(input_ids=input_ids,
                            attention_mask=input_attention_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_padding_mask,
                            past_key_values=None,
                            inputs_embeds=None,
                            use_cache=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=True, )

        logits = outputs["lm_logits"]
        lm_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        values, indices = lm_probs.max(dim=-1)
        seqs = indices.clone().unsqueeze(1)

        loss = values
        counts = 1
        next_x = indices.view(-1, 1)
        input_ids = torch.cat((input_ids, next_x), 1)
        input_attention_mask = torch.cat(
            [input_attention_mask, torch.ones(input_attention_mask.size(0), 1, device=input_attention_mask.device)], 1)

        # Sample from top k

        for _ in range(self.cfg.max_dec_length):
            if history_states and history_attention:
                outputs = model(input_ids=input_ids,
                                attention_mask=input_attention_mask,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=encoder_padding_mask,
                                past_key_values=None,
                                inputs_embeds=None,
                                history_states_list=history_states,
                                history_attention_list=history_attention,
                                use_cache=None,
                                output_attentions=None,
                                output_hidden_states=None,
                                return_dict=True, )
            else:
                outputs = model(input_ids=input_ids,
                                attention_mask=input_attention_mask,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=encoder_padding_mask,
                                past_key_values=None,
                                inputs_embeds=None,
                                use_cache=None,
                                output_attentions=None,
                                output_hidden_states=None,
                                return_dict=True, )
            logits = outputs["lm_logits"]
            lm_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            lm_probs = logits_processor(input_ids, lm_probs)

            # Sample from top k
            values, next_idx = lm_probs.max(dim=-1)

            loss += values
            counts += 1

            next_idx = next_idx.unsqueeze(1)
            seqs = torch.cat([seqs, next_idx], 1)

            if (next_idx.item() == self.cfg.eos_token_id) or (seqs.shape[-1] == (end_len - 1)):
                break

            input_ids, input_attention_mask = self.append_batch(input_ids, next_idx, input_attention_mask)

        beams = self.cfg.tok.batch_decode(seqs)

        sampling_result = {
            "sequence": beams[0],
            "beams": beams,
            "beam_losses": [loss.item()],
            "loss": loss.item(),
            "beam_lengths": [counts],
            "length": counts,
            "hidden_states": outputs["hidden_states"]
        }

        return sampling_result


class TopKSampler(Sampler):
    def __init__(self, cfg):
        super(TopKSampler, self).__init__(cfg)

    def make_batch(self, X):
        X = np.array(X)
        assert X.ndim in [1, 2]
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        batch = torch.tensor(X, dtype=torch.long).to(self.cfg.device)
        return batch

    def generate_sequence(self,
                          batch,
                          model,
                          start_idx,
                          end_len,
                          repetition_penalty=1.0,
                          no_repeat_ngram_size=0,
                          length_penalty=1.0,
                          num_return_sequences=None,
                          top_p=0.7,
                          temperature=None,
                          bart_model=False):

        num_beams = num_return_sequences if num_return_sequences else 1
        logits_processor = self.get_logits_processor(repetition_penalty=repetition_penalty,
                                                     no_repeat_ngram_size=no_repeat_ngram_size)
        logits_warper = self.get_logits_warper(
            top_k=self.cfg.topK, top_p=top_p, temperature=temperature, num_beams=num_beams
        )

        pad_token_id = self.cfg.pad_token_id
        eos_token_id = self.cfg.eos_token_id
        max_length = end_len

        output_scores = False
        output_attentions = False
        output_hidden_states = False

        return_dict_in_generate = False

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids = batch["input_ids"][:, :start_idx]
        input_attention_mask = batch["input_attention_mask"][:, :start_idx]
        encoder_hidden_states = batch["encoder_hidden_states"]
        encoder_padding_mask = batch["encoder_padding_mask"]
        # init sequence length tensors
        batch_size = input_ids.shape[0]
        expanded_return_idx = (
            torch.arange(batch_size).view(-1, 1).repeat(1, num_beams).view(-1).to(encoder_hidden_states.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        input_attention_mask = input_attention_mask.index_select(0, expanded_return_idx)
        encoder_hidden_states = encoder_hidden_states.index_select(0, expanded_return_idx)
        encoder_padding_mask = encoder_padding_mask.index_select(0, expanded_return_idx)

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = init_sequence_length_for_generation(
            input_ids, max_length
        )

        # auto-regressive generation
        while cur_len < end_len:
            outputs = model(input_ids=input_ids,
                            attention_mask=input_attention_mask,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_padding_mask,
                            past_key_values=None,
                            inputs_embeds=None,
                            use_cache=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=True, )

            if bart_model:
                logits = model.lm_head(outputs[0]) + model.final_logits_bias
                past_key_values = None
                hidden_states = None
            else:
                logits = outputs["lm_logits"]
                past_key_values = outputs["past_key_values"]
                hidden_states = outputs['hidden_states']

            next_token_logits = logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (outputs.decoder_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (outputs.decoder_hidden_states,)

            # sample
            lm_probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(lm_probs, num_samples=1).squeeze(1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

        beams = self.cfg.tok.batch_decode(input_ids, skip_special_tokens=True)
        if num_beams and num_beams > 1:
            beams = ['|'.join(beams[num_beams * i: num_beams * (i + 1)]) for i in range(batch_size)]

        # update model kwargs
        past_key_values = outputs["past_key_values"]
        hidden_states = outputs['hidden_states']
        sampling_result = {
            "output_ids": input_ids,
            "beams": beams,
            "hidden_states": hidden_states,
            "past_key_values": past_key_values
        }

        return sampling_result


class BeamSampler(Sampler):
    def __init__(self, cfg):
        super(BeamSampler, self).__init__(cfg)

    def make_batch(self, X):
        X = np.array(X)
        assert X.ndim in [1, 2]
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        batch = torch.tensor(X, dtype=torch.long).to(self.cfg.device)
        return batch

    def append_batch(self, X, beam_toks, mask):
        next_x = beam_toks.unsqueeze(1).unsqueeze(1)
        next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
        return torch.cat((X, next_x), 1), next_mask

    def generate_sequence(self,
                          batch,
                          model,
                          start_idx,
                          end_len,
                          repetition_penalty=1.0,
                          no_repeat_ngram_size=0,
                          length_penalty=1.0,
                          early_stopping=True,
                          num_return_sequences=1,
                          bart_model=False):
        logits_processor = self.get_logits_processor(repetition_penalty=repetition_penalty,
                                                     no_repeat_ngram_size=no_repeat_ngram_size)
        input_ids = batch["input_ids"][:, :start_idx]
        input_attention_mask = batch["input_attention_mask"][:, :start_idx]
        encoder_hidden_states = batch["encoder_hidden_states"]
        encoder_padding_mask = batch["encoder_padding_mask"]

        batch_size = input_ids.shape[0]
        num_beams = self.cfg.eval_bs
        device = input_ids.device

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=end_len,
            num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )

        expanded_return_idx = (
            torch.arange(batch_size).view(-1, 1).repeat(1, num_beams).view(-1).to(encoder_hidden_states.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        input_attention_mask = input_attention_mask.index_select(0, expanded_return_idx)
        encoder_hidden_states = encoder_hidden_states.index_select(0, expanded_return_idx)
        encoder_padding_mask = encoder_padding_mask.index_select(0, expanded_return_idx)

        # print(f"decoder_input_ids: {decoder_input_ids.shape}")
        batch_beam_size, cur_len = input_ids.shape

        assert (
                num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        for _ in range(end_len - 1):
            # Compute distribution for current beam
            if bart_model:
                decoder = model.get_decoder()
            else:
                decoder = model

            outputs = decoder(input_ids=input_ids,
                              attention_mask=input_attention_mask,
                              encoder_hidden_states=encoder_hidden_states,
                              encoder_attention_mask=encoder_padding_mask,
                              past_key_values=None,
                              inputs_embeds=None,
                              use_cache=None,
                              output_attentions=None,
                              output_hidden_states=None,
                              return_dict=True, )

            if bart_model:
                logits = model.lm_head(outputs[0]) + model.final_logits_bias
                past_key_values = None
                hidden_states = None
            else:
                logits = outputs["lm_logits"]
                past_key_values = outputs["past_key_values"]
                hidden_states = outputs['hidden_states']

            next_token_logits = logits[:, -1, :]
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=end_len
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.cfg.pad_token_id,
                eos_token_id=self.cfg.eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=self.cfg.pad_token_id,
            eos_token_id=self.cfg.eos_token_id
        )
        # print(f"decoded:{decoded}")
        seqs = decoded["sequences"]
        beams = self.cfg.tok.batch_decode(seqs, skip_special_tokens=True)

        sampling_result = {
            "output_ids": seqs,
            "beams": beams,
            "hidden_states": hidden_states,
            "past_key_values": past_key_values
        }

        return sampling_result

    def adjust_logits_during_generation(self, logits: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to adjust the logits in
        the generate method.
        """
        return logits

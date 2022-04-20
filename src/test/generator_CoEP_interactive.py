import torch
from src.common.tools import model_device
from configs.basic_config import config
from src.callback.progressbar import ProgressBar

relations = config['relations']


class Generator(object):
    def __init__(self, model, tok, logger, n_gpu, input_len=None):
        self.model = model
        self.tok = tok
        self.logger = logger
        self.input_len = input_len
        self.model, self.device = model_device(n_gpu=n_gpu, model=self.model)

    def generate_example(self,
                         data,
                         sampler,
                         max_length,
                         repetition_penalty=None,
                         no_repeat_ngram_size=None,
                         length_penalty=None,
                         num_return_sequences=1):

        pbar = ProgressBar(n_total=len(data), desc='Testing')
        print_interval = len(data) // 10

        all_generated_sents = []
        all_input_sents = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids = batch

                # [bsz, ln]
                batch_size, ln = input_ids.shape

                char_relation = self.tok(relations, padding=True, return_length=True, return_tensors='pt')
                relation_ids = char_relation["input_ids"].to(input_ids.device)
                relation_mask = char_relation["attention_mask"].to(input_ids.device)

                dim = relation_ids.size(0)

                kg_input_ids = []
                for i in range(batch_size):
                    cur_kg_input_ids = []
                    cur_input_ids = input_ids[i]
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

                # Encoder the story
                encoder_outputs = self.model.encoder(input_ids=input_ids,
                                                     input_masks=input_mask,
                                                     segment_ids=segment_ids,
                                                     kg_input_ids=kg_input_ids,
                                                     kg_input_masks=kg_input_masks,
                                                     )

                encoder_hidden_states = encoder_outputs["encoder_hidden_states"]
                encoder_attentions = encoder_outputs["encoder_attention_mask"]

                decoder_input_ids = torch.ones([batch_size, 1], dtype=torch.long) * self.tok.bos_token_id
                decoder_input_ids = decoder_input_ids.to(self.device)
                batch_data = {}
                batch_data["input_ids"] = decoder_input_ids
                batch_data["input_attention_mask"] = torch.ones_like(decoder_input_ids, dtype=torch.long)
                batch_data["encoder_hidden_states"] = encoder_hidden_states
                batch_data["encoder_padding_mask"] = encoder_attentions

                decoder = self.model.decoder

                sampling_result = sampler.generate_sequence(batch_data,
                                                            decoder,
                                                            start_idx=1,
                                                            end_len=max_length,
                                                            repetition_penalty=repetition_penalty,
                                                            no_repeat_ngram_size=no_repeat_ngram_size,
                                                            length_penalty=length_penalty,
                                                            num_return_sequences=num_return_sequences)
                seqs = sampling_result["beams"]
                # decoder_hidden_states = sampling_result["hidden_states"]
                all_generated_sents += seqs

                input_sent = self.tok.batch_decode(input_ids, skip_special_tokens=True)
                all_input_sents += input_sent

                if (step + 1) % max(1, print_interval) == 0:
                    show_info = pbar(step=step)
                    self.logger.info(show_info)

            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

        return all_generated_sents

    def interactive_generate_example(self,
                                     data,
                                     sampler,
                                     max_length,
                                     repetition_penalty=None,
                                     no_repeat_ngram_size=None,
                                     length_penalty=None,
                                     num_return_sequences=1):

        with torch.no_grad():
            input_ids = data['input_ids'].to(self.device)
            input_masks = data['input_masks'].to(self.device)
            segment_ids = data['segment_ids'].to(self.device)

            char_relation = self.tok(relations, padding=True, return_length=True, return_tensors='pt')
            relation_ids = char_relation["input_ids"].to(self.device)
            relation_mask = char_relation["attention_mask"].to(self.device)

            dim = relation_ids.size(0)
            ln = input_ids.size(0)

            cur_kg_input_ids = []
            cur_input_ids = input_ids
            cur_ln = input_masks.sum()
            for j in range(dim):
                char_ln = relation_mask[j].sum()
                cur_relation_ids = relation_ids[j]
                if cur_ln + char_ln > ln:
                    it = torch.cat([cur_input_ids[:ln - char_ln - 1], cur_input_ids[cur_ln - 1:cur_ln],
                                    cur_relation_ids[:char_ln]], dim=-1).unsqueeze(0)
                    cur_kg_input_ids.append(it)
                else:
                    it = torch.cat(
                        [cur_input_ids[:cur_ln], cur_relation_ids[:char_ln], cur_input_ids[cur_ln + char_ln:]],
                        dim=-1).unsqueeze(0)
                    cur_kg_input_ids.append(it)

            kg_input_ids = torch.cat(cur_kg_input_ids, dim=0).unsqueeze(0).to(self.device)
            kg_input_masks = kg_input_ids.ne(self.tok.pad_token_id).int().to(self.device)

            input_ids = input_ids.unsqueeze(0)
            input_masks = input_masks.unsqueeze(0)
            segment_ids = segment_ids.unsqueeze(0)

            # Encoder the story
            encoder_outputs = self.model.encoder(input_ids=input_ids,
                                                 input_masks=input_masks,
                                                 segment_ids=segment_ids,
                                                 kg_input_ids=kg_input_ids,
                                                 kg_input_masks=kg_input_masks)

            encoder_hidden_states = encoder_outputs["encoder_hidden_states"]
            encoder_attentions = encoder_outputs["encoder_attention_mask"]

            decoder_input_ids = torch.ones([1, 1], dtype=torch.long) * self.tok.bos_token_id
            decoder_input_ids = decoder_input_ids.to(self.device)
            batch_data = {}
            batch_data["input_ids"] = decoder_input_ids
            batch_data["input_attention_mask"] = torch.ones_like(decoder_input_ids, dtype=torch.long)
            batch_data["encoder_hidden_states"] = encoder_hidden_states
            batch_data["encoder_padding_mask"] = encoder_attentions

            decoder = self.model.decoder

            sampling_result = sampler.generate_sequence(batch_data,
                                                        decoder,
                                                        start_idx=1,
                                                        end_len=max_length,
                                                        repetition_penalty=repetition_penalty,
                                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                                        length_penalty=length_penalty,
                                                        num_return_sequences=num_return_sequences
                                                        )
            seqs = sampling_result["beams"]
            input_sent = self.tok.batch_decode(input_ids, skip_special_tokens=True)

            output_ids = sampling_result["output_ids"]
            output_mask = (1 - output_ids.eq(self.tok.pad_token_id).int()).to(self.device)

            decoder_outputs = decoder(input_ids=output_ids,
                                      attention_mask=output_mask,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=encoder_attentions,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      use_cache=None,
                                      output_attentions=True,
                                      output_hidden_states=None,
                                      return_dict=True, )

            attentions = decoder_outputs["attentions"]  # tensor [bsz, self.num_heads, tgt_len, src_len]
            cross_attentions = decoder_outputs["cross_attentions"]  # tensor [bsz, self.num_heads, tgt_len, src_len]

            data = {"input_ids": input_ids,
                    "input_mask": input_masks,
                    "output_ids": output_ids,
                    "encoder_attentions": encoder_attentions,
                    "self_attention": attentions,
                    "cross_attentions": cross_attentions
                    }

            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

        return data, input_sent, seqs

    def interactive_generate_explanation(self,
                                         data,
                                         KG_model,
                                         sampler,
                                         max_length,
                                         repetition_penalty=None,
                                         no_repeat_ngram_size=None,
                                         length_penalty=None,
                                         num_return_sequences=1):

        with torch.no_grad():
            input_ids = data['input_ids']
            input_masks = data['input_masks']
            segment_ids = data['segment_ids']

            # [bsz, ln]
            batch_size, ln = input_ids.shape

            char_relation = self.tok(relations, padding=True, return_length=True, return_tensors='pt')
            relation_ids = char_relation["input_ids"].to(input_ids.device)
            relation_mask = char_relation["attention_mask"].to(input_ids.device)

            dim = relation_ids.size(0)

            kg_input_ids = []
            for i in range(batch_size):
                cur_kg_input_ids = []
                cur_input_ids = input_ids[i]
                cur_ln = input_masks[i].sum()
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
            kg_input_ids = torch.cat(kg_input_ids, dim=0).to(self.device)
            kg_input_masks = kg_input_ids.ne(self.tok.pad_token_id).int().to(self.device)

            # Encoder the story
            encoder = KG_model.get_encoder().to(self.device)
            encoder_outputs = encoder(input_ids=kg_input_ids.reshape([batch_size * dim, ln]),
                                      attention_mask=kg_input_masks.reshape([batch_size * dim, ln]),
                                      inputs_embeds=None,
                                      output_attentions=None,
                                      output_hidden_states=None,
                                      return_dict=None, )
            KG_hidden_states = encoder_outputs[0]
            bsz, dim, ln = kg_input_masks.size()

            decoder_input_ids = torch.ones([batch_size * dim, 1], dtype=torch.long) * self.tok.bos_token_id
            decoder_input_ids = decoder_input_ids.to(self.device)
            batch_data = {}
            batch_data["input_ids"] = decoder_input_ids
            batch_data["input_attention_mask"] = torch.ones_like(decoder_input_ids, dtype=torch.long).to(
                self.device)
            batch_data["encoder_hidden_states"] = KG_hidden_states.reshape([batch_size * dim, ln, -1]).to(
                self.device)
            batch_data["encoder_padding_mask"] = kg_input_masks.reshape([batch_size * dim, ln]).to(self.device)

            decoder = KG_model.to(self.device)

            sampling_result = sampler.generate_sequence(batch_data,
                                                        decoder,
                                                        start_idx=1,
                                                        end_len=max_length,
                                                        repetition_penalty=repetition_penalty,
                                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                                        length_penalty=length_penalty,
                                                        bart_model=True,
                                                        num_return_sequences=num_return_sequences)
            seqs = sampling_result["beams"]

            # decoder_hidden_states = sampling_result["hidden_states"]
            dim = dim * num_return_sequences

            seqs_list = []
            for j in range(int(dim / num_return_sequences)):
                seqs_list.append(
                    '|'.join(seqs[j * num_return_sequences: (j + 1) * num_return_sequences]))

            input_sent = self.tok.batch_decode(input_ids, skip_special_tokens=True)

            output_ids = sampling_result["output_ids"]
            assert output_ids.size(0) == batch_size * dim

            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

        return input_sent, seqs_list

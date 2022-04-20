import pandas as pd
import torch
from src.common.tools import model_device
from src.callback.progressbar import ProgressBar
from configs.basic_config import config


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
                         relation_index=None,
                         repetition_penalty=None,
                         no_repeat_ngram_size=None,
                         length_penalty=None,
                         num_return_sequences=1,
                         save_prefix=None,
                         add_cls=False):

        pbar = ProgressBar(n_total=len(data), desc='Testing')
        print_interval = len(data) // 10

        all_generated_sents = []
        all_label_sents = []
        all_input_sents = []
        all_encoder_states = []
        all_cls_ids = []
        all_self_attention = []
        all_cross_attentions = []
        all_encoder_attentions = []
        all_input_ids = []
        all_output_ids = []
        all_input_mask = []

        relations = config['relations']

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, labels_masks, cls_label = batch

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

                if relation_index is not None:
                    encoder_attentions[:, :dim] = 0
                    encoder_attentions[:, relation_index] = 1

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

                label_sent = self.tok.batch_decode(label_ids, skip_special_tokens=True)
                all_label_sents += label_sent

                input_sent = self.tok.batch_decode(input_ids, skip_special_tokens=True)
                all_input_sents += input_sent

                all_encoder_states.append(encoder_hidden_states)

                output_ids = sampling_result["output_ids"]
                assert output_ids.size(0) == batch_size
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

                attentions = decoder_outputs["attentions"]  # list of [bsz, self.num_heads, tgt_len, src_len]
                cross_attentions = decoder_outputs[
                    "cross_attentions"]  # list of [bsz, self.num_heads, tgt_len, src_len]

                # print(f"attention from last decoder layer: {attentions[-1].size()}")
                # print(f"cross attention from last encoder layer: {cross_attentions[-1].size()}")
                # print(f"attention from last encoder layer: {encoder_attentions[-1].size()}")
                all_encoder_attentions.append(encoder_attentions[0].cpu().detach())
                all_self_attention.append(attentions[0].cpu().detach())
                all_cross_attentions.append(cross_attentions[0].cpu().detach())
                all_input_ids.append(input_ids.cpu().detach())
                all_output_ids.append(output_ids.cpu().detach())
                all_input_mask.append(input_mask.cpu().detach())

                if add_cls:
                    decoder_hidden_states = decoder_outputs["hidden_states"]  # [bsz, ln, E]
                    eos_hidden_states = []
                    eos_index = torch.sum(output_mask, dim=1) - 2  # [bsz]
                    batch_size = eos_index.size(0)
                    for i in range(batch_size):
                        eos_hidden_states.append(decoder_hidden_states[i, eos_index[i], :])
                    eos_hidden_states = torch.cat([x.unsqueeze(0) for x in eos_hidden_states])
                    cls_logits = self.model.classification_head(eos_hidden_states)  # [bsz, num_labels]
                    cls_ids = torch.argmax(cls_logits.softmax(dim=-1), dim=-1)
                    all_cls_ids.append(cls_ids)

                if (step + 1) % max(1, print_interval) == 0:
                    show_info = pbar(step=step)
                    self.logger.info(show_info)

            data = {"input_ids": all_input_ids,
                    "input_mask": all_input_mask,
                    "output_ids": all_output_ids,
                    "encoder_attentions": all_encoder_attentions,
                    "self_attention": all_self_attention,
                    "cross_attentions": all_cross_attentions
                    }

            # file_path = f"test_attentions_layer_0_{self.args.data_name}.pkl"
            # import pickle
            # with open(file_path, 'wb') as f:
            #     pickle.dump(data, f)

            if save_prefix is not None:
                columns = ["input", "generated", "reference"]
                # print(f"all_generated_sents: {len(all_generated_sents)} -- all_label_sents: {len(all_label_sents)}")
                df = pd.DataFrame(columns=columns)
                df["input"] = all_input_sents
                df["generated"] = all_generated_sents
                df["reference"] = all_label_sents
                if len(all_cls_ids) != 0:
                    df["cls_id"] = torch.cat(all_cls_ids, dim=0).cpu().detach().tolist()
                df.to_csv(f"{save_prefix}")

                self.logger.info(f"save sentences in csv files: {save_prefix}. Done!")

            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

        return all_label_sents, all_generated_sents, all_encoder_states, all_cls_ids

    def generate_explanation(self,
                             data,
                             KG_model,
                             sampler,
                             max_length,
                             relations=None,
                             repetition_penalty=None,
                             no_repeat_ngram_size=None,
                             length_penalty=None,
                             save_prefix=None,
                             KG_only=False,
                             num_return_sequences=1):

        pbar = ProgressBar(n_total=len(data), desc='Testing')
        print_interval = len(data) // 10

        all_generated_sents = []
        all_label_sents = []
        all_input_sents = []
        all_encoder_states = []
        all_cls_ids = []

        relations = relations if relations is not None else config['relations']

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, labels_masks, labels_segment_ids, cls_label = batch

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
                kg_input_ids = torch.cat(kg_input_ids, dim=0).to(self.device)
                kg_input_masks = kg_input_ids.ne(self.tok.pad_token_id).int().to(self.device)

                # Encoder the story
                if KG_only:
                    encoder = KG_model.get_encoder().to(self.device)
                    encoder_outputs = encoder(input_ids=kg_input_ids.reshape([batch_size * dim, ln]),
                                              attention_mask=kg_input_masks.reshape([batch_size * dim, ln]),
                                              inputs_embeds=None,
                                              output_attentions=None,
                                              output_hidden_states=None,
                                              return_dict=None, )
                    KG_hidden_states = encoder_outputs[0]
                else:
                    encoder_outputs = self.model.encoder(input_ids=input_ids,
                                                         input_masks=input_mask,
                                                         segment_ids=segment_ids,
                                                         kg_input_ids=kg_input_ids,
                                                         kg_input_masks=kg_input_masks,
                                                         )

                    KG_hidden_states = encoder_outputs["kg_hidden_states"]  # [bsz, dim, ln, E]

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
                seqs_results = []
                for i in range(batch_size):
                    batch_seqs = seqs[i * dim:(i + 1) * dim]
                    batch_seqs_list = []
                    for j in range(int(dim / num_return_sequences)):
                        batch_seqs_list.append(
                            '|'.join(batch_seqs[j * num_return_sequences: (j + 1) * num_return_sequences]))
                    # print(f"batch_seqs_list: {len(batch_seqs_list)},\n {batch_seqs_list}")
                    seqs_results.append(batch_seqs_list)
                all_generated_sents += seqs_results

                input_sent = self.tok.batch_decode(input_ids, skip_special_tokens=True)
                all_input_sents += input_sent

                output_ids = sampling_result["output_ids"]
                assert output_ids.size(0) == batch_size * dim

                if (step + 1) % max(1, print_interval) == 0:
                    show_info = pbar(step=step)
                    self.logger.info(show_info)

            if save_prefix is not None:
                if "csv" in save_prefix:
                    df = pd.DataFrame(all_generated_sents, columns=relations)
                    df.to_csv(f"{save_prefix}")
                else:
                    assert "pkl" in save_prefix
                    import pickle
                    with open(save_prefix, 'wb') as f:
                        pickle.dump({"input": all_input_sents,
                                     "generated": all_generated_sents}, f)

                self.logger.info(f"save sentences in csv files: {save_prefix}. Done!")

            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

        return all_label_sents, all_generated_sents, all_encoder_states, all_cls_ids

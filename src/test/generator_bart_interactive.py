import torch
import pandas as pd
from src.common.tools import model_device
from src.callback.progressbar import ProgressBar
from src.train.metrics import Accuracy, F1Score


class Generator(object):
    def __init__(self, model, tok, logger, n_gpu):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu=n_gpu, model=self.model)
        self.tok = tok

    def interactive_generate(self, data, none_token_id=39763):
        pbar = ProgressBar(n_total=len(data), desc='Testing')
        print_interval = len(data) // 10
        all_logits = []
        all_labels = []
        accuracy = Accuracy(topK=1)
        f1 = F1Score(task_type="multiclass", average="macro")
        for step, batch in enumerate(data):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, labels_masks = batch
                _, cls_logits = self.model(input_ids,
                                           attention_mask=input_mask,
                                           decoder_input_ids=None,
                                           labels=label_ids,
                                           output_attentions=False,
                                           output_hidden_states=False,
                                           return_dict=True)
                cls_lables = []
                for i, label in enumerate(label_ids):
                    if none_token_id in label:
                        cls_lables.append(0)
                    else:
                        cls_lables.append(1)
                cls_lables = torch.tensor(cls_lables, dtype=torch.long).to(self.device)
                all_logits.append(cls_logits)
                all_labels.append(cls_lables)
            accuracy(logits=cls_logits, target=cls_lables)
            f1(logits=cls_logits, target=cls_lables)
            info = {accuracy.name(): accuracy.value(), f1.name(): f1.value()}
            if (step + 1) % max(1, print_interval) == 0:
                show_info = pbar(step=step, info=info)
                self.logger.info(show_info)

        logits = torch.cat(all_logits, dim=0)
        target = torch.cat(all_labels, dim=0).contiguous().view(-1)
        accuracy(logits=logits, target=target)
        f1(logits=logits, target=target)
        info = {accuracy.name(): accuracy.value(), f1.name(): f1.value()}
        show_info = "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])

        probs = torch.softmax(logits, dim=1)
        indice = torch.argmax(probs, dim=1)

        self.logger.info(f"save sentences in csv files:test_case_cls.csv")

        cols = ["predicted_refs", "test_refs"]
        df = pd.DataFrame(columns=cols)
        df["predicted_refs"] = indice.cpu().numpy().tolist()
        df["test_refs"] = target.cpu().numpy().tolist()
        df.to_csv("test_case_cls.csv", index=True)
        self.logger.info(show_info)

    def generate(self,
                 data,
                 max_length,
                 num_beams,
                 num_return_sequences=1,
                 length_penalty=None,
                 early_stopping=True,
                 do_sample=False):
        pbar = ProgressBar(n_total=len(data), desc='Testing')
        print_interval = len(data) // 10
        all_input_sents = []
        all_generated_sents = []
        for step, batch in enumerate(data):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids = batch
                output_ids = self.model.generate(input_ids,
                                                 max_length=max_length,
                                                 num_return_sequences=num_return_sequences,
                                                 num_beams=num_beams,
                                                 length_penalty=length_penalty,
                                                 early_stopping=early_stopping,
                                                 do_sample=do_sample)
            generated_sent = self.tok.batch_decode(output_ids, skip_special_tokens=True)
            batch_input_strs = self.tok.batch_decode(input_ids, skip_special_tokens=True)
            all_input_sents += batch_input_strs
            all_generated_sents += generated_sent
            if (step + 1) % max(1, print_interval) == 0:
                show_info = pbar(step=step)
                self.logger.info(show_info)

        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return all_generated_sents

    def generate_with_cls(self,
                          data,
                          sampler,
                          max_length,
                          repetition_penalty=None,
                          no_repeat_ngram_size=None,
                          length_penalty=None,
                          save_prefix=None,
                          num_return_sequences=1,
                          add_cls=True):

        pbar = ProgressBar(n_total=len(data), desc='Testing')
        print_interval = len(data) // 10

        all_generated_sents, all_label_sents = [], []
        all_input_sents = []
        all_encoder_states = []
        all_encoder_attentions = []
        all_input_ids = []
        all_input_mask = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, labels_masks, labels_segment_ids, cls_label = batch

                # [bsz, ln]
                batch_size, s_ln = input_ids.shape

                encoder = self.model.get_encoder()
                # Encoder the story

                encoder_outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    output_attentions=True
                )

                encoder_hidden_states = encoder_outputs.last_hidden_state
                encoder_attentions = encoder_outputs.attentions

                decoder_input_ids = torch.ones([batch_size, 1], dtype=torch.long) * self.tok.bos_token_id
                decoder_input_ids = decoder_input_ids.to(self.device)
                batch_data = {}
                batch_data["input_ids"] = decoder_input_ids
                batch_data["input_attention_mask"] = torch.ones_like(decoder_input_ids, dtype=torch.long)
                batch_data["encoder_hidden_states"] = encoder_hidden_states
                batch_data["encoder_padding_mask"] = input_mask

                sampling_result = sampler.generate_sequence(batch_data,
                                                            model=self.model,
                                                            start_idx=1,
                                                            end_len=max_length,
                                                            repetition_penalty=repetition_penalty,
                                                            no_repeat_ngram_size=no_repeat_ngram_size,
                                                            length_penalty=length_penalty,
                                                            num_return_sequences=num_return_sequences,
                                                            bart_model=True)
                seqs = sampling_result["beams"]
                # decoder_hidden_states = sampling_result["hidden_states"]
                for i in range(batch_size):
                    s = seqs[i * num_return_sequences: (i + 1) * num_return_sequences]
                    all_generated_sents.append(s)

                label_sent = self.tok.batch_decode(label_ids, skip_special_tokens=True)
                all_label_sents += label_sent

                input_sent = self.tok.batch_decode(input_ids, skip_special_tokens=True)
                all_input_sents += input_sent

                all_encoder_states.append(encoder_hidden_states)
                all_encoder_attentions.append(encoder_attentions[0].cpu().detach())
                all_input_ids.append(input_ids.cpu().detach())
                all_input_mask.append(input_mask.cpu().detach())

                if (step + 1) % max(1, print_interval) == 0:
                    show_info = pbar(step=step)
                    self.logger.info(show_info)

            if save_prefix is not None:
                if "csv" in save_prefix:
                    columns = ["input", "generated", "reference"]
                    df = pd.DataFrame(columns=columns)
                    df["input"] = all_input_sents
                    df["generated"] = all_generated_sents
                    df["reference"] = all_label_sents
                    df.to_csv(f"{save_prefix}")
                else:
                    assert "pkl" in save_prefix
                    import pickle
                    with open(save_prefix, 'wb') as f:
                        pickle.dump({"input": all_input_sents,
                                     "generated": all_generated_sents,
                                     "reference": all_label_sents}, f)

                self.logger.info(f"save sentences in csv files: {save_prefix}. Done!")

            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

        return all_label_sents, all_generated_sents, all_encoder_states

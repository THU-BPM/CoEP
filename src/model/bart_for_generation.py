import torch.nn as nn
import torch
from transformers import BartModel, BartForConditionalGeneration


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_classes: int,
            pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartWithClassification(nn.Module):
    def __init__(self, path, num_labels):
        super(BartWithClassification, self).__init__()

        self.num_labels = num_labels
        self.model = BartModel.from_pretrained(path)

        self.classification_head = BartClassificationHead(
            self.model.config.d_model,
            self.model.config.d_model,
            self.num_labels,
            self.model.config.classifier_dropout,
        )

        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(self.model.config.d_model, self.model.shared.num_embeddings, bias=False)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             inputs_embeds=inputs_embeds,
                             decoder_inputs_embeds=decoder_inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict
                             )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        cls_logits = None

        if decoder_input_ids is not None:

            decoder_hidden_states = outputs.last_hidden_state
            batch_size = decoder_hidden_states.size(0)

            eos_mask = decoder_input_ids.eq(self.model.config.eos_token_id)
            eos_mask[eos_mask.sum(1).eq(0), -1] = True
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            eos_hidden_states = decoder_hidden_states[eos_mask, :].view(
                decoder_hidden_states.size(0), -1, decoder_hidden_states.size(-1))[:, -1, :]  # [bsz, E]
            cls_logits = self.classification_head(eos_hidden_states)  # [bsz, num_labels]
            assert eos_hidden_states.dim() == 2
            assert eos_hidden_states.shape[0] == batch_size

        return {"lm_logits": lm_logits,
                "cls_logits": cls_logits}

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_cls_head(self):
        return self.classification_head



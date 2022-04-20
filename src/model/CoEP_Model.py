import torch
import torch.nn as nn

from .utils.modeling import BartClassificationHead, BartEncoder, BartDecoder


class Encoder(nn.Module):
    def __init__(self,
                 bart_config,
                 im_encoder_sates=None,
                 gm_encoder_states=None,
                 fixed_kg=False):
        # bart_config cannot be None
        super(Encoder, self).__init__()
        self.input_size = bart_config.d_model
        self.dropout = bart_config.dropout

        # use fine-tuned kg model encoder, and freeze the embedding of kg component in case of kg model decoding
        self.im_encoder = BartEncoder(config=bart_config, )
        if im_encoder_sates:
            self.im_encoder.load_state_dict(im_encoder_sates)
            if fixed_kg:
                self.im_encoder.requires_grad_(False)

        self.gm_encoder = BartEncoder(config=bart_config, )
        if gm_encoder_states:
            self.gm_encoder.load_state_dict(gm_encoder_states)

    def forward(self,
                input_ids,
                input_masks,
                segment_ids,
                kg_input_ids,
                kg_input_masks,
                output_attentions=None):
        """
        @param input_ids: [bsz, ln]
        @param input_masks: [bsz, ln]
        @param segment_ids: [bsz, ln]
        @param kg_input_ids: [bsz, dim, ln]
        @param kg_input_masks: [bsz, dim, ln]
        @param output_attentions: bool
        @return:
        """
        attentions = None

        kg_batch_size, kg_ln = kg_input_ids.size()

        kg_hidden_states = self.im_encoder.forward(input_ids=kg_input_ids,
                                                   attention_mask=kg_input_masks,
                                                   output_attentions=True)['last_hidden_state']

        eos_indice = torch.zeros(size=[kg_batch_size, kg_ln], dtype=torch.int).to(kg_hidden_states.device)
        eos_indice = eos_indice.scatter(1, (kg_input_masks.sum(dim=1) - 1).unsqueeze(dim=1), 1)
        eos_indice = eos_indice.unsqueeze(2).expand(kg_hidden_states.size()).bool()
        # bool tensor [batch_size, ln] where True indicate the position of eos
        kg_re = kg_hidden_states.masked_select(eos_indice).view([kg_batch_size, kg_hidden_states.size(-1)])

        batch_size = input_ids.size(0)
        dim = kg_batch_size / batch_size

        kg_re = kg_re.reshape(batch_size, dim, -1)  # (batch_size, dim, E)

        context_hidden_states = self.gm_encoder.forward(input_ids=input_ids,
                                                        attention_mask=input_masks)['last_hidden_state']
        # has shape [batch_size, ln, E]

        encoder_hidden_states = torch.cat([kg_re, context_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([torch.ones([batch_size, dim], dtype=torch.int).to(input_masks.device),
                                            input_masks], dim=1)
        # [batch_size, ln+dim, E]

        return {"encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
                "kg_hidden_states": kg_hidden_states.reshape([batch_size, dim, kg_ln, -1]),
                "attentions": attentions}


class Decoder(nn.Module):
    def __init__(self, gm_decoder_states, bart_config=None):
        super(Decoder, self).__init__()
        self.input_size = bart_config.d_model
        self.vocab_size = bart_config.vocab_size

        self.gm_decoder = BartDecoder(config=bart_config)
        if gm_decoder_states:
            self.gm_decoder.load_state_dict(gm_decoder_states)
        self.lm_head = nn.Linear(self.input_size, self.vocab_size)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        """

        @param input_ids:
        @param attention_mask:
        @param encoder_hidden_states:
        @param encoder_attention_mask:
        @param past_key_values:
        @param inputs_embeds:
        @param use_cache:
        @param output_attentions:
        @param output_hidden_states:
        @param return_dict:
        @return: lm_logits: [bsz, ln, vocab_size]
        """
        outputs = self.gm_decoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        hidden_states = outputs["last_hidden_state"]  # [bsz, t_ln, E]
        past_key_values = outputs["past_key_values"]
        all_attentions = outputs["attentions"] if output_attentions else None
        all_cross_attentions = outputs["cross_attentions"] if output_attentions else None
        lm_logits = self.lm_head(hidden_states)

        return {"lm_logits": lm_logits,
                "hidden_states": hidden_states,
                "past_key_values": past_key_values,
                "attentions": all_attentions,
                "cross_attentions": all_cross_attentions
                }


class CoEP(nn.Module):  # context encoder but only decoder once
    """
        train single decoder as sentence level model
    """

    def __init__(self, bart_config, pretrained_im=None, pretrained_gm=None, num_labels=None, fixed_kg=False):
        super(CoEP, self).__init__()
        self.bart_config = bart_config
        self.pad_token_id = self.bart_config.pad_token_id
        self.input_size = self.bart_config.d_model
        self.vocab_size = self.bart_config.vocab_size
        self.dropout = self.bart_config.dropout

        self.num_labels = num_labels
        im_encoder_states = pretrained_im.get_encoder().state_dict() if pretrained_im else None
        gm_encoder_states = pretrained_gm.get_encoder().state_dict() if pretrained_gm else None
        gm_decoder_states = pretrained_gm.get_decoder().state_dict() if pretrained_gm else None

        self.encoder = Encoder(bart_config,
                               im_encoder_sates=im_encoder_states,
                               gm_encoder_states=gm_encoder_states,
                               fixed_kg=fixed_kg)

        self.decoder = Decoder(gm_decoder_states=gm_decoder_states,
                               bart_config=bart_config)

        if self.num_labels is not None:
            self.classification_head = BartClassificationHead(input_dim=self.input_size,
                                                              inner_dim=self.input_size,
                                                              num_classes=self.num_labels,
                                                              pooler_dropout=self.dropout)

    def forward(self,
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                im_input_ids,
                im_input_masks,
                segment_ids=None,
                labels=None,
                use_cache=False,
                past_key_values=None,
                return_dict=None
                ):
        if past_key_values is not None:
            use_cache = True

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            input_masks=attention_mask,
            segment_ids=segment_ids,
            kg_input_ids=im_input_ids,
            kg_input_masks=im_input_masks,
        )

        encoder_hidden_states = encoder_outputs["encoder_hidden_states"]
        encoder_attention_mask = encoder_outputs["encoder_attention_mask"]
        kg_hidden_states = encoder_outputs["kg_hidden_states"]

        # Decoder the motiv
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=use_cache,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )

        batch_size = decoder_input_ids.size(0)
        cls_logits = None
        if self.num_labels is not None:
            decoder_hidden_states = decoder_outputs["hidden_states"]  # [bsz, ln, E]
            eos_mask = decoder_input_ids.eq(self.bart_config.eos_token_id)
            eos_mask[eos_mask.sum(1).eq(0), -1] = True
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            eos_hidden_states = decoder_hidden_states[eos_mask, :].view(
                decoder_hidden_states.size(0), -1, decoder_hidden_states.size(-1))[:, -1, :]  # [bsz, E]
            cls_logits = self.classification_head(eos_hidden_states)  # [bsz, num_labels]

        lm_logits = decoder_outputs["lm_logits"]
        decoder_hidden_states = decoder_outputs["hidden_states"]
        decoder_past_key_values = decoder_outputs["past_key_values"]

        return {"lm_logits": lm_logits,
                "cls_logits": cls_logits,
                "kg_hidden_states": kg_hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "decoder_past_key_values": decoder_past_key_values,
                "decoder_hidden_states": decoder_hidden_states}

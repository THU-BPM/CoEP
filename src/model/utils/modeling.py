import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
from torch import Tensor
import torch.nn.functional as F
import math
import numpy as np
import random
from transformers import BartConfig, BartPretrainedModel
from .util import make_causal_mask, expand_mask, ACT2FN


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


class Attention(nn.Module):
    def __init__(
            self,
            embed_dim,
            dropout=0.0,
            bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = nn.Linear(embed_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self,
            inputs: Optional[Tensor],
            padding_mask: Optional[Tensor] = None,
            output_attentions=False,
    ):
        """

        :param inputs: [bsz, ln, E]
        :param padding_mask: [bsz, ln], 1: token, 0: padding
        :param output_attentions: bool
        :return: outputs [bsz, E]
                attn_weights [bsz, ln]

        """
        proj_x = self.proj(inputs)  # [bs, len, E]
        v_x = self.v(proj_x).squeeze(-1)  # [bs, len]
        attn_weights = self.softmax(self.tanh(v_x))  # [bs, len]
        if padding_mask is not None:  # don't attend to padding symbols
            reshaped = padding_mask.eq(0)
            attn_weights = attn_weights.masked_fill(reshaped, 0.)

        outputs = torch.matmul(
            inputs.permute(0, 2, 1),
            attn_weights.unsqueeze(2)
        ).squeeze(2)  # [bsz, embed_dim]
        if output_attentions:
            return outputs, attn_weights
        else:
            return outputs


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper. from bart_modeling"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Gate(nn.Module):
    def __init__(self, embed_dim, dropout=0.0, bias=True):
        super(Gate, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.bias = bias

        self.linear_r = nn.Linear(2 * embed_dim, 1, bias=self.bias)
        self.linear_z = nn.Linear(2 * embed_dim, 1, bias=self.bias)
        self.linear_c = nn.Linear(2 * embed_dim, embed_dim, bias=self.bias)

    def forward(self,
                history_feature,
                input_feature,
                output_gates=False
                ):
        """
        :param history_feature: [tgt_len, bsz, embed_dim]
        :param input_feature: [tgt-len, bsz, embed_dim]
        :return:
        """
        h = torch.cat([input_feature, history_feature], dim=-1)  # [tgt_len, bsz, 2 * embde_dim]
        r = self.linear_r(h).sigmoid()  # [tgt_len, bsz, 1]
        z = self.linear_z(h).sigmoid()  # [tgt_len, bsz, 1]
        h_t = torch.matmul(history_feature.unsqueeze(dim=3), r.unsqueeze(dim=3)).squeeze()  # [tgt_len, bsz, embed_dim]
        h_new = self.linear_c(torch.cat([input_feature, h_t], dim=-1))  # [tgt_len, bsz, embe_dim]
        ones = torch.ones_like(z)
        output = torch.matmul(h_new.unsqueeze(dim=3), z.unsqueeze(dim=3)) + \
                 torch.matmul(history_feature.unsqueeze(dim=3), (ones - z).unsqueeze(dim=3))
        output = output.squeeze()  # [tgt_len, bsz, embed_dim]
        return output


class Fusion(nn.Module):
    def __init__(self, hidden_size, dropout=0.0, bias=True):
        super(Fusion, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bias = bias

        self.linear_r = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=self.bias)
        self.linear_g = nn.Linear(2 * self.hidden_size, 1, bias=self.bias)

    def forward(self,
                history_feature,
                inputs):
        """
        history_feature: [bsz, ln, E]
        inputs: [bsz, ln, E]
        """
        r = self.linear_r(torch.cat([history_feature, inputs], dim=-1)).tanh()  # [bsz, ln, E]
        g = self.linear_g(torch.cat([history_feature, inputs], dim=-1)).sigmoid()  # [bsz, ln, 1]
        ones = torch.ones_like(g)
        # print(f"-------------r: {r.size()}--------------")
        # print(f"-------------g: {g.size()}--------------")
        expand = [1] * inputs.dim()
        expand[-1] = inputs.shape[-1]
        output = torch.mul(r, g.repeat(expand)) + torch.mul(inputs, (ones - g).repeat(expand))  # [bsz, ln, E]
        # the same size as inputs
        return output


class BartEncoderLayer(nn.Module):
    def __init__(self, bart_config):
        super().__init__()
        self.embed_dim = bart_config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=bart_config.encoder_attention_heads,
            dropout=bart_config.dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = bart_config.dropout
        self.activation_fn = ACT2FN[bart_config.activation_function]
        self.activation_dropout = bart_config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, bart_config.encoder_ffn_dim)
        self.fc2 = nn.Linear(bart_config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayer(nn.Module):
    def __init__(self, bart_config):
        super().__init__()
        self.embed_dim = bart_config.d_model
        self.num_heads = bart_config.decoder_attention_heads
        self.dropout = bart_config.dropout

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            is_decoder=True,
        )
        self.activation_fn = ACT2FN[bart_config.activation_function]
        self.activation_dropout = bart_config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            bart_config.decoder_attention_heads,
            dropout=bart_config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, bart_config.decoder_ffn_dim)
        self.fc2 = nn.Linear(bart_config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super(BartEncoder).__init__(config)

        self.config = config

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, output_attentions=output_attentions)

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return {
            "last_hidden_state": hidden_states, "hidden_states": encoder_states, "attentions": all_attentions}


class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def forward(
        self,
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
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                input_shape, inputs_embeds.dtype, tgt_len=input_shape[-1], past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None and combined_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            combined_attention_mask = combined_attention_mask + expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False):
                if use_cache:
                    raise ValueError(
                        "When using `gradient_checkpointing, make sure that `use_cache=False` and `config.use_cache=False`."
                    )

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "cross_attentions":all_cross_attentions,}


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long,
                                 device=self.weight.device)
        return super().forward(positions)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, "`padding_idx` should not be None, but of type int"
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)
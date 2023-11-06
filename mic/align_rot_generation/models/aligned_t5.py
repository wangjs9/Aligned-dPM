# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model."""

import warnings

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


def RankingLoss(score, gold_score=None, margin=0.001, gold_margin=0, gold_weight=0, no_gold=False, no_cand=False):
    loss_func = torch.nn.MarginRankingLoss(0.0, reduction="sum")
    loss_mask = (score != 0).long()
    TotalLoss = loss_func(score, score, loss_mask) / loss_mask.sum()
    # candidate loss
    n = score.size(1)
    no_cand = True if score.sum() == 0 else no_cand
    if not no_cand:
        for i in range(1, n):
            pos_score, neg_score = score[:, :-i], score[:, i:]
            pos_score, neg_score = pos_score.contiguous().view(-1), neg_score.contiguous().view(-1)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='sum')
            loss_mask = ((pos_score != 0) & (neg_score != 0)).long()
            if loss_mask.sum() == 0:
                continue
            total_pair = neg_score.size(0)
            extra_margin = (total_pair - loss_mask.sum()) * margin * i
            loss = (loss_func(pos_score, neg_score, loss_mask) - extra_margin) / loss_mask.sum()
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold response loss
    if gold_weight > 0:
        pos_score = gold_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        loss_func = torch.nn.MarginRankingLoss(gold_margin, reduce='sum')
        loss_mask = (neg_score != 0).long()
        if loss_mask.sum() == 0:
            return TotalLoss
        TotalLoss += gold_weight * loss_func(pos_score, neg_score, loss_mask) / loss_mask.sum()
    return TotalLoss


class AlignedT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            score_mode="base",
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache != None else self.config.use_cache
        return_dict = return_dict if return_dict != None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask != None and decoder_head_mask == None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs == None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        batch_size, candidate_labels = None, None
        if labels != None and len(labels.size()) > 2:
            batch_size = input_ids.size(0)
            candidate_labels = labels.clone()
            labels = candidate_labels[:, 0].contiguous()
            candidate_num = candidate_labels.size(1)
            hidden_states = torch.repeat_interleave(hidden_states, candidate_num, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, candidate_num, dim=0)

        if labels != None and decoder_input_ids == None and decoder_inputs_embeds == None:
            # get decoder inputs from shifting lm labels to the right
            if candidate_labels != None:
                decoder_input_ids = self._shift_right(candidate_labels.view(batch_size * candidate_num, labels.size(2)))
            else:
                decoder_input_ids = self._shift_right(labels)
        elif candidate_labels != None:
            decoder_input_ids = decoder_input_ids.view(batch_size * candidate_num, decoder_input_ids.size(2))

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids != None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask != None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask != None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        masked_lm_loss, ranking_loss = None, None
        if candidate_labels != None:
            all_lm_logits = lm_logits.view(batch_size, candidate_num, lm_logits.size(1), lm_logits.size(2))
            lm_logits = all_lm_logits[:, 0].contiguous()
            if score_mode == "log":
                _all_lm_logits = F.log_softmax(all_lm_logits, dim=-1)
            else:
                _all_lm_logits = F.softmax(all_lm_logits, dim=-1)
            candidate_mask = candidate_labels != -100  # this is 0
            scores = torch.gather(_all_lm_logits, 3,
                                  candidate_labels.masked_fill(~candidate_mask, 0).unsqueeze(-1)).squeeze(-1)
            candidate_mask = candidate_mask.float()
            scores = torch.mul(scores, candidate_mask).sum(-1) / (candidate_mask.sum(-1) + 1e-6)
            golden_similarity, candidate_similarity = scores[:, 0], scores[:, 1:]
            ranking_loss = RankingLoss(candidate_similarity, golden_similarity)

        if labels != None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if ranking_loss != None:
            # loss = masked_lm_loss + ranking_loss
            loss = masked_lm_loss + 0.72 * ranking_loss
        else:
            loss = masked_lm_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss != None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

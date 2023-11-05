# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BART model."""
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BartForConditionalGeneration, BartConfig, BartModel
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput


def RankingLoss(score, gold_score=None, margin=0.001, gold_margin=0, gold_weight=0, no_gold=False, no_cand=False):
    if score.sum() == 0:
        return 0
    loss_func = torch.nn.MarginRankingLoss(0.0, reduction="sum")
    loss_mask = (score != 0).long()
    TotalLoss = loss_func(score, score, loss_mask) / loss_mask.sum()
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='sum')
            loss_mask = ((pos_score != 0) & (neg_score != 0)).long()
            if loss_mask.sum() == 0:
                continue
            pos_score = pos_score * loss_mask
            neg_score = neg_score * loss_mask
            total_pair = neg_score.size(0) * neg_score.size(1)
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


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id == None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class AlignedBartModel(BartModel):

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
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids == None and decoder_inputs_embeds == None:
            if input_ids == None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions != None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states != None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache != None else self.config.use_cache
        return_dict = return_dict if return_dict != None else self.config.use_return_dict

        if encoder_outputs == None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_batch_size = decoder_input_ids.size(0) if decoder_input_ids != None else decoder_inputs_embeds.size(0)
        hidden_states = None
        if decoder_batch_size != encoder_outputs[0].size(0):
            candidate_num = decoder_batch_size // input_ids.size(0)
            hidden_states = torch.repeat_interleave(encoder_outputs[0], candidate_num, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, candidate_num, dim=0)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states if hidden_states != None else encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    batch_size, sequence_length = input_ids.size(0), input_ids.size(-1)
    candidate_num = -1
    if len(input_ids.size()) == 3:
        candidate_num = input_ids.size(1)
        input_ids = input_ids.view(batch_size * candidate_num, sequence_length)
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    if candidate_num > 0:
        shifted_input_ids = shifted_input_ids.view(batch_size, candidate_num, sequence_length)

    return shifted_input_ids


class AlignedBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = AlignedBartModel(config)

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
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict != None else self.config.use_return_dict
        batch_size = input_ids.size(0) if input_ids != None else encoder_outputs[0].size(0)
        if labels != None:
            if len(labels.size()) > 2:
                candidate_labels = labels.clone()
                candidate_num = candidate_labels.size(1)
                if decoder_input_ids == None:
                    decoder_input_ids = shift_tokens_right(
                        candidate_labels.view(batch_size * candidate_num, candidate_labels.size(-1)),
                        self.config.pad_token_id,
                        self.config.decoder_start_token_id,
                    )
                else:
                    decoder_input_ids = decoder_input_ids.view(batch_size * candidate_num, decoder_input_ids.size(-1))

            elif decoder_input_ids == None and decoder_inputs_embeds == None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss, ranking_loss = None, None
        if lm_logits.size(0) > batch_size:
            all_lm_logits = lm_logits.view(batch_size, -1, lm_logits.size(1), lm_logits.size(2))
            # [bz, candidate_num, seq_len, word_dim]
            lm_logits = all_lm_logits[:, 0].contiguous()
            labels = candidate_labels[:, 0].contiguous()
            if score_mode == "log":
                _all_lm_logits = F.log_softmax(all_lm_logits, dim=-1)
            else:
                _all_lm_logits = F.softmax(all_lm_logits, dim=-1)
            candidate_mask = (candidate_labels != -100) & (candidate_labels != 0)  # this is 0
            scores = torch.gather(_all_lm_logits, 3,
                                  candidate_labels.masked_fill(~candidate_mask, 0).unsqueeze(-1)).squeeze(-1)
            candidate_mask = candidate_mask.float()
            scores = torch.mul(scores, candidate_mask).sum(-1) / (candidate_mask.sum(-1) + 1e-6)
            golden_similarity, candidate_similarity = scores[:, 0], scores[:, 1:]
            ranking_loss = RankingLoss(candidate_similarity, golden_similarity)

        if labels != None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if ranking_loss != None:
            # loss = masked_lm_loss + ranking_loss
            loss = masked_lm_loss + 0.72 * ranking_loss
        else:
            loss = masked_lm_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss != None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

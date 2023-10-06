import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForSequenceClassification
from typing import Optional
try:
    from preference_modeling.models.common import BaseModel, PrefixBertModel
except ModuleNotFoundError:
    from models.common import BaseModel, PrefixBertModel

class Model(BaseModel, BertForSequenceClassification):
    def __init__(self, config: BertConfig, args):
        super().__init__(config)
        self.n_embd = self.config.hidden_size
        self.mid_dim = args.d_prefix
        self.match_n_layer = self.config.num_hidden_layers // 2
        self.match_n_head = self.config.num_attention_heads
        self.match_n_embd = self.config.hidden_size // self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.class_num = args.class_num

        # The Multi prefix modules!
        # The task-prefix modules from all specific tasks
        self.prefix_names = ["prediction"]
        self.preseqlen = args.preseqlen
        self.prefix_tokens = torch.arange(self.preseqlen).long()
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.multi_prefix = nn.ModuleDict(
            {
                name: nn.ModuleDict(
                    {
                        "wte_enc": nn.Embedding(self.preseqlen, self.n_embd),
                        "control_trans_enc": nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.Tanh(),
                            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
                        ),
                        "mlp": nn.Sequential(
                            nn.Linear(config.hidden_size, self.mid_dim),
                            nn.Tanh(),  # nn.LeakyReLU(),
                            nn.Linear(self.mid_dim, self.class_num),
                        ),
                    }
                )
                for name in self.prefix_names
            }
        )
        self.prefix_dropout = nn.Dropout(args.prefix_dropout)
        self.bert = PrefixBertModel(config)
        self.classifier = None

    def get_label(self, labels, bsz=None):
        """ Convert labels to majority labels.

        :param labels: (bsz, )
        :param bsz: batch size
        :return major_labels: (bsz, )

        """
        class_num = self.class_num
        labels_dist = torch.zeros(bsz, class_num + 1).to(labels.device)
        one_tensor = torch.ones_like(labels_dist)
        label_mask = labels == -100
        labels = labels.masked_fill(label_mask, class_num)
        labels_dist = labels_dist.scatter_add(1, labels, one_tensor)
        labels_dist = labels_dist[:, :-1]
        labels_dist[:, -1] = labels_dist[:, -1] + 0.5
        major_labels = labels_dist.argmax(-1)
        assert major_labels.shape[0] == bsz
        return major_labels

    def get_prompt(self, name, bsz=None, device="cuda"):
        old_bsz = bsz

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)
        )
        temp_control_enc = self.multi_prefix[name]["wte_enc"](input_tokens_enc)
        temp_control_enc = self.prefix_dropout(temp_control_enc)
        past_key_values_enc = self.multi_prefix[name]["control_trans_enc"](
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        past_key_values_enc = self.prefix_dropout(past_key_values_enc)
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val_enc in enumerate(past_key_values_enc):
            temp = dict()
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(device).bool(),
            }
            result.append(temp)

        return result

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            scores: Optional[torch.Tensor] = None,
            validation=False,
            **kwargs,
    ):
        assert self.toker is not None

        bsz = input_ids.size(0)
        distribution = {}
        past_prompt = self.get_prompt("prediction", bsz, input_ids.device)
        outputs = self.bert(
            input_ids,
            past_prompt,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )[0]
        masks = input_ids == 0 if token_type_ids == None else token_type_ids == 0
        pooler = outputs.masked_fill(masks.unsqueeze(-1), 0)
        pooler = pooler.sum(dim=1) / (1 - masks.long()).sum(dim=1).unsqueeze(-1)
        distribution["prediction"] = self.multi_prefix["prediction"]["mlp"](pooler).softmax(-1)

        loss = None
        if labels is not None:
            distribution["major_label"] = self.get_label(labels, bsz)
            loss = self.comp_loss(distribution)
        res = {"dist": distribution["prediction"], "loss": loss}
        if not self.training and not validation:
            return res
        else:
            assert not validation
        return res

    ### compute loss
    def comp_loss(self, distribution):
        prediction, major_label = distribution["prediction"], distribution["major_label"]
        loss_ce = nn.CrossEntropyLoss(reduction="mean")
        l = loss_ce(prediction, major_label)
        return l

    @torch.no_grad()
    def predict(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        assert not self.training
        assert self.toker is not None

        bsz = input_ids.size(0)
        past_prompt = self.get_prompt("prediction", bsz, input_ids.device)
        outputs = self.bert(
            input_ids,
            past_prompt,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )[0]

        masks = token_type_ids == 0
        pooler = outputs.masked_fill(masks.unsqueeze(-1), 0)
        pooler = pooler.sum(dim=1) / (1 - masks.long()).sum(dim=1).unsqueeze(-1)
        distribution = self.multi_prefix["prediction"]["mlp"](pooler).softmax(-1)

        return distribution

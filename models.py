import transformers
import torch
import torch.nn as nn

from transformers import AlbertModel, AlbertConfig, AlbertPreTrainedModel


class RSAHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.activation = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.max_position_embeddings // 2))  # output is just bits long, not 2 * bits
        self.decoder = nn.Linear(config.embedding_size, config.max_position_embeddings // 2)

        self.out_activation = nn.Sigmoid()

        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = self.out_activation(hidden_states)

        return prediction_scores


class AlbertForDiffieHellman(AlbertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.classifier = RSAHead(config)
        self.loss = nn.L1Loss()  # use l1 loss

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # transformer step
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # get pooled output
        sequence_output = outputs[1]

        # logits and loss
        logits = self.classifier(sequence_output)
        loss = self.loss(logits, labels)

        return loss, logits

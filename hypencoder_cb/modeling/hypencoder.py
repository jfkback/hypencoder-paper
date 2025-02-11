import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from hypencoder_cb.modeling.q_net import RepeatedDenseBlockConverter
from hypencoder_cb.modeling.similarity_and_losses import (
    HypencoderCrossEntropyLoss,
    HypencoderMarginMSELoss,
)


class EncoderOutput(ModelOutput):
    representation: Any
    loss: Optional[torch.Tensor] = None


@dataclass
class DualEncoderOutput(ModelOutput):
    query_output: Optional[EncoderOutput] = None
    passage_output: Optional[EncoderOutput] = None
    similarity: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    to_log: Optional[Dict] = None


class BaseDualEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        query_encoder_type: str = "",
        passage_encoder_type: str = "",
        query_encoder_kwargs: Dict = {},
        passage_encoder_kwargs: Dict = {},
        loss_type: Union[str, List[str]] = "",
        loss_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = {},
        shared_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(loss_type, str):
            loss_type = [loss_type]

        if isinstance(loss_kwargs, dict):
            loss_kwargs = [loss_kwargs]

        assert len(loss_type) == len(loss_kwargs)

        self.query_encoder_type = query_encoder_type
        self.passage_encoder_type = passage_encoder_type
        self.query_encoder_kwargs = query_encoder_kwargs
        self.passage_encoder_kwargs = passage_encoder_kwargs
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs
        self.shared_encoder = shared_encoder


class BaseDualEncoder(PreTrainedModel):
    config_class = BaseDualEncoderConfig

    def __init__(self, config: BaseDualEncoderConfig):
        super(BaseDualEncoder, self).__init__(config)
        self._get_similarity_loss(config)
        self.similarity_loss_forward_kwargs = [
            {} for _ in range(len(self.similarity_losses))
        ]

    def _get_similarity_loss(self, config: BaseDualEncoderConfig):
        raise NotImplementedError()

    def _get_encoder_losses(
        self,
        output: EncoderOutput,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = output.representation.device

        if output.loss is not None:
            return output.loss
        else:
            return torch.tensor(0.0, device=device)

    def forward(
        self,
        query_input_ids: Optional[torch.LongTensor] = None,
        query_attention_mask: Optional[torch.LongTensor] = None,
        passage_input_ids: Optional[torch.LongTensor] = None,
        passage_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        query_input_kwargs: Optional[Dict] = None,
        passage_input_kwargs: Optional[Dict] = None,
        full_output: bool = False,
    ) -> DualEncoderOutput:
        if query_input_kwargs is None:
            query_input_kwargs = {}

        if passage_input_kwargs is None:
            passage_input_kwargs = {}

        if query_input_ids is None and passage_input_ids is None:
            raise ValueError(
                "At least one of query_input_ids or passage_input_ids"
                " must be provided"
            )

        query_output = None
        if query_input_ids is not None:
            query_output = self.query_encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                **query_input_kwargs,
            )

        passage_output = None
        if passage_input_ids is not None:
            passage_output = self.passage_encoder(
                input_ids=passage_input_ids,
                attention_mask=passage_attention_mask,
                **passage_input_kwargs,
            )

        output = DualEncoderOutput(
            query_output=query_output,
            passage_output=passage_output,
        )

        if self.training or full_output:
            to_log = {}

            total_similarity_loss = torch.tensor(0.0, device=self.device)
            for i, similarity_loss in enumerate(self.similarity_losses):
                similarity_loss_output = similarity_loss(
                    query_output,
                    passage_output,
                    labels=labels,
                    **self.similarity_loss_forward_kwargs[i],
                )

                total_similarity_loss += similarity_loss_output.loss

                if len(self.similarity_losses) > 1:
                    to_log[f"similarity_loss_{self.config.loss_type[i]}"] = (
                        similarity_loss_output.loss.item()
                    )

            loss = total_similarity_loss

            query_encoder_loss = self._get_encoder_losses(
                query_output, device=query_input_ids.device
            )
            passage_encoder_loss = self._get_encoder_losses(
                passage_output, device=passage_input_ids.device
            )

            to_log["similarity_loss"] = loss.item()
            to_log["query_encoder_loss"] = query_encoder_loss.item()
            to_log["passage_encoder_loss"] = passage_encoder_loss.item()

            loss += query_encoder_loss
            loss += passage_encoder_loss

            output = DualEncoderOutput(
                query_output=query_output,
                passage_output=passage_output,
                loss=loss,
                similarity=similarity_loss_output.similarity,
                to_log=to_log,
            )

        return output


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dim: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    score = torch.einsum("bqd,bkd->bqk", query, key) / math.sqrt(dim)

    if mask is not None:
        score.masked_fill_(mask.unsqueeze(1) == 0, -float("Inf"))

    attention = F.softmax(score, -1)

    context = torch.einsum("bqk,bkd->bqd", [attention, value])
    return context, attention


class HypencoderConfig(PretrainedConfig):
    def __init__(
        self,
        model_name_or_path: str = "",
        freeze_transformer: bool = False,
        converter_kwargs: Dict = {},
        embedding_representation: Optional[str] = None,
        base_encoder_output_dim: int = 768,
        **kwargs,
    ):
        """
        Args:
            model_name_or_path (str, optional): The base transformer model to
                use. Defaults to "".
            freeze_transformer (bool, optional): Whether to freeze the base
                transformer model. Defaults to False.
            converter_kwargs (Dict, optional): Key-word arguments passed to the
                q-net converter function. Defaults to {}.
            embedding_representation (Optional[str], optional): If provided
                the model's output will include an embedding representations
                as well as the q-net representations. The options are "mean"
                and "cls". Defaults to None.
            base_encoder_output_dim (int, optional): The output dimension of
                the base transformer model. Defaults to 768.
        """

        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.freeze_transformer = freeze_transformer
        self.converter_kwargs = converter_kwargs
        self.embedding_representation = embedding_representation
        self.base_encoder_output_dim = base_encoder_output_dim


@dataclass
class HypencoderOutput(EncoderOutput):
    embedding_representation: Optional[torch.Tensor] = None


class Hypencoder(PreTrainedModel):
    config_class = HypencoderConfig

    def __init__(self, config: HypencoderConfig) -> None:
        super(Hypencoder, self).__init__(config)
        self.transformer = AutoModel.from_pretrained(config.model_name_or_path)
        self.weight_to_model_converter = RepeatedDenseBlockConverter(
            **config.converter_kwargs
        )

        if config.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.weight_shapes = self.weight_to_model_converter.weight_shapes
        self.bias_shapes = self.weight_to_model_converter.bias_shapes

        self._initialize_hyper_head()

    def _initialize_hyper_head(self) -> None:
        torch.manual_seed(1)

        model_dim = self.config.base_encoder_output_dim

        self.hyper_base_matrices = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(1, out_dim, in_dim, requires_grad=True),
                    requires_grad=True,
                )
                for in_dim, out_dim in self.weight_shapes
            ]
        )

        self.hyper_base_vectors = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(out_dim, in_dim, requires_grad=True),
                    requires_grad=True,
                )
                for in_dim, out_dim in self.bias_shapes
            ]
        )

        self.weight_query_embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(1, out_dim, in_dim, requires_grad=True),
                    requires_grad=True,
                )
                for in_dim, out_dim in self.weight_shapes
            ]
        )

        self.bias_query_embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(1, out_dim, in_dim, requires_grad=True),
                    requires_grad=True,
                )
                for in_dim, out_dim in self.bias_shapes
            ]
        )

        self.weight_hyper_projection = nn.ParameterList(
            [
                nn.Linear(in_dim, in_dim)
                for in_dim, out_dim in self.weight_shapes
            ]
        )

        self.bias_hyper_projection = nn.ParameterList(
            [nn.Linear(in_dim, in_dim) for in_dim, out_dim in self.bias_shapes]
        )

        self.key_projections = nn.ParameterList(
            [
                nn.Linear(model_dim, in_dim)
                for in_dim, out_dim in (self.weight_shapes + self.bias_shapes)
            ]
        )

        self.value_projections = nn.ParameterList(
            [
                nn.Linear(model_dim, in_dim)
                for in_dim, out_dim in (self.weight_shapes + self.bias_shapes)
            ]
        )

        with torch.no_grad():
            for i in range(len(self.weight_shapes)):
                nn.init.normal_(
                    self.hyper_base_matrices[i].data, std=(2 / 768) ** 0.5
                )
                nn.init.normal_(
                    self.weight_query_embeddings[i].data, mean=0, std=10
                )
                nn.init.normal_(
                    self.weight_hyper_projection[i].weight, std=(1 / (768**2))
                )
                nn.init.zeros_(self.weight_hyper_projection[i].bias)

                # The vectors linear layer biases are not zero'd in this
                # implementation which is mostly an oversight.
                nn.init.zeros_(self.key_projections[i].bias)
                nn.init.zeros_(self.value_projections[i].bias)

            for i in range(len(self.bias_shapes)):
                nn.init.zeros_(self.hyper_base_vectors[i].data)
                nn.init.normal_(
                    self.bias_query_embeddings[i].data, mean=0, std=10
                )

    def _get_weights_and_biases(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """From the last hidden state of the transformer model and the
        attention mask, compute the weights and biases for the q-net.

        Args:
            last_hidden_state (torch.Tensor): Shape (bs, seq_len, hidden_size)
            attention_mask (torch.Tensor): Shape (bs, seq_len)

        Returns:
            (Tuple[List[torch.Tensor], List[torch.Tensor]]): The weights and
                biases for the q-nets. The weights are a list of tensors with
                shape (bs, out_dim, in_dim) and the biases are a list of
                tensors with shape (bs, out_dim). The order of the weights and
                biases is the same as the order of the weight and bias shapes
                provided by the converter.
        """

        batch_size = last_hidden_state.size(0)

        # Shape (num_hyper_layers, batch_size, max_seq_size, hidden_size)
        keys = [
            key_projection(last_hidden_state)
            for key_projection in self.key_projections
        ]
        values = [
            value_projection(last_hidden_state)
            for value_projection in self.value_projections
        ]

        weights = []
        for i in range(len(self.weight_shapes)):
            weights.append(
                scaled_dot_product_attention(
                    query=self.weight_query_embeddings[i].repeat_interleave(
                        batch_size, dim=0
                    ),
                    key=keys[i],
                    value=values[i],
                    dim=self.weight_shapes[i][1],
                    mask=attention_mask,
                )[0]
            )

        biases = []
        offset = len(self.weight_shapes)
        for i in range(len(self.bias_shapes)):
            biases.append(
                scaled_dot_product_attention(
                    query=self.bias_query_embeddings[i].repeat_interleave(
                        batch_size, dim=0
                    ),
                    key=keys[i + offset],
                    value=values[i + offset],
                    dim=self.bias_shapes[i][1],
                    mask=attention_mask,
                )[0]
            )

        weights_final = []
        biases_final = []

        for i in range(len(self.weight_shapes)):
            weights_final.append(
                self.weight_hyper_projection[i](
                    F.layer_norm(F.relu(weights[i]), weights[i].shape[2:])
                )
            )

        for i in range(len(self.bias_shapes)):
            biases_final.append(
                self.bias_hyper_projection[i](
                    F.layer_norm(F.relu(biases[i]), biases[i].shape[2:])
                )
            )

        weights_final = [
            (
                weights_final[i]
                + self.hyper_base_matrices[i].repeat(batch_size, 1, 1)
            ).transpose(2, 1)
            for i in range(len(self.weight_shapes))
        ]

        biases_final = [
            (
                biases_final[i]
                + self.hyper_base_vectors[i].repeat(batch_size, 1, 1)
            ).transpose(2, 1)
            for i in range(len(self.bias_shapes))
        ]

        return weights_final, biases_final

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Shape (bs, seq_len, hidden_size)
        last_hidden_state = output.last_hidden_state

        matrices, vectors = self._get_weights_and_biases(
            last_hidden_state, attention_mask
        )

        models = self.weight_to_model_converter(
            matrices, vectors, is_training=self.training
        )

        output = HypencoderOutput(representation=models)

        if self.config.embedding_representation is not None:
            if self.config.embedding_representation == "mean":
                output.embedding_representation = last_hidden_state.sum(
                    dim=1
                ) / (attention_mask.sum(dim=1, keepdim=True))
            elif self.config.embedding_representation == "cls":
                output.embedding_representation = last_hidden_state[:, 0]
            else:
                raise ValueError("Unknown embedding representation type")

        return output


class TextEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        model_name_or_path: str = "",
        pooling_type: str = "cls",
        freeze_transformer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_name_or_path = model_name_or_path
        self.pooling_type = pooling_type
        self.freeze_transformer = freeze_transformer


class TextEncoder(PreTrainedModel):
    config_class = TextEncoderConfig

    def __init__(self, config: TextEncoderConfig) -> None:
        super(TextEncoder, self).__init__(config)
        self.transformer = AutoModel.from_pretrained(config.model_name_or_path)
        self.pooling_type = config.pooling_type

        if config.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        if self.pooling_type == "mean":
            self.pool = self.mean_pool
        elif self.pooling_type == "cls":
            self.pool = self.cls_pool

    def mean_pool(self, last_hidden_state, attention_mask):
        return last_hidden_state.sum(dim=1) / attention_mask.sum(
            dim=1, keepdim=True
        )

    def cls_pool(self, last_hidden_state, attention_mask):
        return last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask):
        output = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )

        pooled_output = self.pool(output.last_hidden_state, attention_mask)

        if self.config.include_linear_layer:
            pooled_output = self.linear_layer(pooled_output)

        if self.config.include_layer_norm:
            pooled_output = self.layer_norm(pooled_output)

        return EncoderOutput(representation=pooled_output)


class HypencoderDualEncoderConfig(BaseDualEncoderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class HypencoderDualEncoder(BaseDualEncoder):
    config_class = HypencoderDualEncoderConfig

    def __init__(self, config: HypencoderDualEncoderConfig):
        super(HypencoderDualEncoder, self).__init__(config)

        self.query_encoder = Hypencoder(
            HypencoderConfig(**config.query_encoder_kwargs)
        )

        self.passage_encoder = TextEncoder(
            TextEncoderConfig(**config.passage_encoder_kwargs)
        )

        if config.shared_encoder:
            self.passage_encoder.transformer = self.query_encoder.transformer

    def _get_similarity_loss(self, config: BaseDualEncoderConfig):
        self.similarity_losses = []

        for loss_type, loss_kwargs in zip(
            config.loss_type, config.loss_kwargs
        ):
            if loss_type == "margin_mse":
                self.similarity_losses.append(
                    HypencoderMarginMSELoss(**loss_kwargs)
                )
            elif loss_type == "cross_entropy":
                self.similarity_losses.append(
                    HypencoderCrossEntropyLoss(**loss_kwargs)
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")


class TextDualEncoder(BaseDualEncoder):
    config_class = BaseDualEncoderConfig

    def __init__(self, config: BaseDualEncoderConfig):
        super(TextDualEncoder, self).__init__(config)

        self.query_encoder = TextEncoder(
            TextEncoderConfig(**config.query_encoder_kwargs)
        )

        self.passage_encoder = TextEncoder(
            TextEncoderConfig(**config.passage_encoder_kwargs)
        )

        if config.shared_encoder:
            self.passage_encoder = self.query_encoder

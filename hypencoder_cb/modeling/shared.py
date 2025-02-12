from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
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

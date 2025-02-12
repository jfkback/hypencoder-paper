from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from hypencoder_cb.modeling.shared import EncoderOutput


def pos_neg_triplets_from_similarity(similarity: torch.Tensor) -> torch.Tensor:
    """Takes a similarity matrix and turns it into a matrix of
    positive-negative pairs.

    Args:
        similarity (torch.Tensor): A similarity matrix with shape:
                (num_queries, num_items_per_query).
            It is assumed that the first item in each row is the positive item.

    Returns:
        torch.Tensor: A matrix of positive-negative pairs with shape:
            (num_queries * num_negatives_per_query, 2).
    """

    num_queries, num_items_per_query = similarity.shape
    num_negatives_per_query = num_items_per_query - 1

    if num_items_per_query == 2:
        return similarity

    assert num_items_per_query > 2

    positives = similarity[:, 0]

    output = torch.zeros(
        num_queries * num_negatives_per_query, 2, device=similarity.device
    )
    output[:, 0] = positives.repeat_interleave(num_negatives_per_query)

    for i in range(num_queries):
        output[
            i * num_negatives_per_query : (i + 1) * num_negatives_per_query, 1
        ] = similarity[i, 1:]

    return output


def no_in_batch_negatives_hypecoder_similarity(
    query_models: Callable,
    item_embeddings: torch.Tensor,
    required_num_items_per_query: Optional[int] = None,
) -> torch.Tensor:
    """Takes a set of query models and a set of item embeddings and returns
    the similarity between each query and each item.

    Args:
        query_models (Callable): A callable that takes a tensor of items and
            returns a tensor of similarities.
        item_embeddings (torch.Tensor): A tensor of item embeddings with shape:
            (num_items, item_emb_dim).
        required_num_items_per_query (Optional[int], optional): An optional
            integer that specifies the number of items required per query.
            Defaults to None.

    Returns:
        torch.Tensor: A tensor of similarities with shape:
            (num_queries, num_items_per_query).
    """

    assert len(item_embeddings.shape) == 2

    num_items, item_emb_dim = item_embeddings.shape
    num_queries = query_models.num_queries

    assert num_items % num_queries == 0

    num_items_per_query = num_items // num_queries

    if required_num_items_per_query is not None:
        assert num_items_per_query == required_num_items_per_query

    item_embeddings = item_embeddings.view(
        num_queries, num_items_per_query, item_emb_dim
    )

    similarity = query_models(item_embeddings).squeeze()

    return similarity


def in_batch_negatives_hypecoder_similarity(
    query_models: Callable,
    item_embeddings: torch.Tensor,
    required_num_items_per_query: Optional[int] = None,
) -> torch.Tensor:
    """Takes a set of query models and a set of item embeddings and returns
    the similarity between each query and each item.

    Args:
        query_models (Callable): A callable that takes a tensor of items and
            returns a tensor of similarities.
        item_embeddings (torch.Tensor): A tensor of item embeddings with shape:
            (num_items, item_emb_dim).
        required_num_items_per_query (Optional[int], optional): An optional
            integer that specifies the number of items required per query.
            Defaults to None.

    Returns:
        torch.Tensor: A tensor of similarities with shape:
            (num_queries, num_items).
    """

    assert len(item_embeddings.shape) == 2

    num_items, item_emb_dim = item_embeddings.shape
    num_queries = query_models.num_queries

    item_embeddings = item_embeddings.unsqueeze(0).repeat(num_queries, 1, 1)

    similarity = (
        query_models(item_embeddings).view(num_queries, num_items).squeeze()
    )

    return similarity


@dataclass
class SimilarityAndLossOutput:
    similarity: torch.Tensor
    loss: torch.Tensor


class SimilarityAndLossBase(nn.Module):
    def __init__(self, *args, scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _loss(
        self, similarity: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SimilarityAndLossOutput:
        similarity = self._get_similarity(
            query_output, passage_output, **kwargs
        )
        loss = self.scale * self._loss(similarity, labels, **kwargs)

        return SimilarityAndLossOutput(similarity=similarity, loss=loss)


class MarginMSELoss(SimilarityAndLossBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.MSELoss()

    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _loss(
        self, similarity: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        num_similarity_queries, num_similarity_items = similarity.shape
        num_label_queries, num_label_items = labels.shape

        assert num_similarity_items == 2
        assert num_label_items > 1

        if num_label_items != 2:
            labels = pos_neg_triplets_from_similarity(labels)
            num_label_queries, num_label_items = labels.shape

        assert num_label_items == 2
        assert num_similarity_queries == num_label_queries

        similarity = self.normalization_fn(similarity)
        labels = self.normalization_fn(labels)

        margin = similarity[:, 0] - similarity[:, 1]
        teacher_margin = labels[:, 0] - labels[:, 1]

        return self.loss(margin.squeeze(), teacher_margin.squeeze())


class CrossEntropyLoss(SimilarityAndLossBase):

    def __init__(
        self,
        use_in_batch_negatives: bool = True,
        only_use_first_item: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss = nn.CrossEntropyLoss()

        self.use_in_batch_negatives = use_in_batch_negatives
        self.only_use_first_item = only_use_first_item

    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _loss(
        self, similarity: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.loss(similarity, labels)

    def _get_target(
        self,
        num_queries: int,
        num_items: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_items_per_query = num_items // num_queries

        if self.use_in_batch_negatives:
            targets = torch.arange(
                num_queries,
                dtype=torch.long,
                device=device,
            )
            targets = targets * num_items_per_query

        else:
            targets = torch.zeros(num_queries, dtype=torch.long, device=device)

        return targets

    def forward(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SimilarityAndLossOutput:
        similarity = self._get_similarity(
            query_output, passage_output, **kwargs
        )

        target = self._get_target(
            similarity.size(0), similarity.size(1), device=similarity.device
        )

        loss = self.scale * self._loss(similarity, target, **kwargs)

        return SimilarityAndLossOutput(similarity=similarity, loss=loss)


class HypencoderMarginMSELoss(MarginMSELoss):
    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        similarity = no_in_batch_negatives_hypecoder_similarity(
            query_output.representation,
            passage_output.representation,
        )

        return pos_neg_triplets_from_similarity(similarity)

    def forward(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        labels: Optional[torch.Tensor] = None,
    ) -> SimilarityAndLossOutput:
        loss = torch.tensor(0.0, device=passage_output.representation.device)
        similarity = self._get_similarity(query_output, passage_output)
        loss += self.scale * self._loss(
            similarity,
            labels,
        )

        return SimilarityAndLossOutput(similarity=similarity, loss=loss)


class HypencoderCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
        use_query_embedding_representation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_query_embedding_representation = (
            use_query_embedding_representation
        )

    def _get_similarity(
        self,
        query_output: EncoderOutput,
        passage_output: EncoderOutput,
        **kwargs,
    ) -> torch.Tensor:
        if self.use_in_batch_negatives:
            if self.use_cross_device_negatives:
                raise NotImplementedError(
                    "Cross device negatives not supported for Hypencoder."
                )
            else:
                query_model = query_output.representation
                passage_embeddings = passage_output.representation

            if self.only_use_first_item:
                num_items = passage_embeddings.shape[0]
                num_queries = query_model.num_queries
                items_per_query = num_items // num_queries

                indices = (
                    torch.arange(
                        num_queries,
                        device=passage_embeddings.device,
                        dtype=torch.long,
                    )
                    * items_per_query
                )

                passage_embeddings = passage_embeddings[indices]

            similarity = in_batch_negatives_hypecoder_similarity(
                query_model, passage_embeddings
            )
        else:
            similarity = no_in_batch_negatives_hypecoder_similarity(
                query_output.representation, passage_output.representation
            )

        return similarity

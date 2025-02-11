from typing import List, Optional

import torch
import torch.nn.functional as F


class NoTorchSequential:

    def __init__(
        self,
        layers,
        num_queries: Optional[int] = None,
    ):
        self.layers = layers
        self.num_queries = num_queries

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class NoTorchLinear:
    def __init__(self, weight, bias: Optional[torch.Tensor] = None):
        """
        Args:
            weight (torch.Tensor): Torch tensor that represents the weights
                of the linear function with the shape:
                    (num_queries, input_hidden_size, output_hidden_size)
            bias (Optional[torch.Tensor], optional): Optional torch tensor
                that represents the bias of the linear function with the shape:
                    (num_queries, output_hidden_size).
                Defaults to None.
        """

        self.weight = weight
        self.bias = bias if bias is not None else None

        if self.bias is not None:
            self.bias = self.bias.squeeze().view(-1, 1, weight.shape[-1])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input vectors with the shape:
                (num_queries, num_items_per_query, input_hidden_size)

        Returns:
            torch.Tensor: Output vectors with the shape:
                (num_queries, num_items_per_query, output_hidden_size)
        """
        y = torch.einsum("qin,qnh->qih", x, self.weight)

        if self.bias is not None:
            y += self.bias

        return y


class NoTorchDenseBlock:
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        activation: Optional[torch.nn.Module] = None,
        do_layer_norm: bool = False,
        do_residual: bool = False,
        do_dropout: bool = False,
        dropout_prob: float = 0.1,
        layer_norm_before_residual: bool = True,
    ):
        """
        Args:
            weight (torch.Tensor): The weight matrices with the shape:
                (num_queries, input_hidden_size, output_hidden_size)
            bias (Optional[torch.Tensor], optional): Optional bias vectors
                with the shape:
                    (num_queries, output_hidden_size).
                Defaults to None.
            activation (Optional[torch.nn.Module], optional): The activation
                function to use. Defaults to None.
            do_layer_norm (bool, optional): Whether to apply layer norm.
                Defaults to False.
            do_residual (bool, optional): Whether to apply residual connection.
                Defaults to False.
            do_dropout (bool, optional): Whether to apply dropout. Defaults
                to False.
            dropout_prob (float, optional): The dropout probability. Defaults
                to 0.1.
            layer_norm_before_residual (bool, optional): Whether to apply
                layer norm before residual connection. Defaults to True.
        """

        self.linear = NoTorchLinear(weight, bias)
        self.layer_norm_before_residual = layer_norm_before_residual
        self.activation = activation
        self.do_layer_norm = do_layer_norm
        self.do_residual = do_residual
        self.do_dropout = do_dropout
        self.dropout_prob = dropout_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input vectors with the shape:
                (num_queries, num_items_per_query, input_hidden_size)

        Returns:
            torch.Tensor: Output vectors with the shape:
                (num_queries, num_items_per_query, output_hidden_size)
        """

        if self.do_dropout:
            y = F.dropout(x, self.dropout_prob)
        else:
            y = x.clone()

        y = self.linear(y)

        if self.activation is not None:
            y = self.activation(y)

        if self.do_layer_norm and self.layer_norm_before_residual:
            y = F.layer_norm(y, (y.size(-1),))

        if self.do_residual:
            y = y + x

        if self.do_layer_norm and not self.layer_norm_before_residual:
            y = F.layer_norm(y, (y.size(-1),))

        return y


def activation_factory(
    activation_type: str,
):
    if activation_type == "relu":
        return torch.nn.ReLU()
    elif activation_type == "tanh":
        return torch.nn.Tanh()
    elif activation_type == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation_type == "gelu":
        return torch.nn.GELU()
    elif activation_type == "leaky_relu":
        return torch.nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


class RepeatedDenseBlockConverter:
    def __init__(
        self,
        vector_dimensions: List[int],
        activation_type: str = "relu",
        do_dropout: bool = False,
        dropout_prob: float = 0.1,
        do_layer_norm: bool = True,
        do_residual: bool = True,
        do_residual_on_last: bool = False,
        layer_norm_before_residual: bool = True,
    ):
        """ "
        Args:
            vector_dimensions (List[int]): The dimension of the input,
                intermediate, and output vectors. These are used to determine
                the weight and bias shapes for the q-net. The first value
                should be the dimension of the input vectors, and the last
                value should be 1.
            activation_type (str, optional): The type of activation to use.
                See `activation_factory` for details. Defaults to "relu".
            do_dropout (bool, optional): Whether to apply dropout. Defaults
                to False.
            dropout_prob (float, optional): The dropout probability. Defaults
                to 0.1.
            do_layer_norm (bool, optional): Whether to apply layer norm.
                Defaults to True.
            do_residual (bool, optional): Whether to apply residual connection.
                Defaults to True.
            do_residual_on_last (bool, optional): Whether to apply residual
                connection on the last layer. Defaults to False.
            layer_norm_before_residual (bool, optional): Whether to apply
                layer norm before residual connection. Defaults to True.

        Raises:
            ValueError: If `do_residual_on_last` is True and `do_residual`
                is False.
        """

        self.weight_shapes = []
        for i in range(1, len(vector_dimensions)):
            self.weight_shapes.append(
                (vector_dimensions[i - 1], vector_dimensions[i])
            )

        self.bias_shapes = [
            (vector_dimensions[i], 1)
            for i in range(1, len(vector_dimensions) - 1)
        ]

        self.num_layers = len(self.weight_shapes)

        self.activation = activation_factory(activation_type)
        self.do_dropout = do_dropout
        self.dropout_prob = dropout_prob
        self.do_layer_norm = do_layer_norm
        self.do_residual = do_residual
        self.layer_norm_before_residual = layer_norm_before_residual

        if do_residual_on_last is None:
            do_residual_on_last = do_residual

        self.do_residual_on_last = do_residual_on_last

        if self.do_residual_on_last and not self.do_residual:
            raise ValueError(
                "do_residual_on_last can only be True if do_residual is True."
            )

    def __call__(
        self,
        matrices: List[torch.Tensor],
        vectors: List[torch.Tensor],
        is_training: bool,
    ) -> NoTorchSequential:
        """
        Args:
            matrices (List[torch.Tensor]): The weight matrices with the shapes:
                (num_queries, input_hidden_size, output_hidden_size)
            vectors (List[torch.Tensor]): The bias vectors with the shapes:
                (num_queries, output_hidden_size, 1)
            is_training (bool): Whether the model is in training mode.

        Returns:
            NoTorchSequential: The q-net.
        """

        dropout = self.do_dropout and is_training
        batch_size = matrices[0].size(0)
        num_core_layers = self.num_layers - 1

        layers = []
        for j in range(num_core_layers):
            do_residual = self.do_residual
            if j == num_core_layers - 1 and not self.do_residual_on_last:
                do_residual = False

            weight_matrix = matrices[j]

            if j < len(self.bias_shapes):
                bias_vector = vectors[j]
            else:
                bias_vector = None

            layers.append(
                NoTorchDenseBlock(
                    weight_matrix,
                    bias_vector,
                    activation=self.activation,
                    do_layer_norm=self.do_layer_norm,
                    do_residual=do_residual,
                    do_dropout=dropout,
                    dropout_prob=self.dropout_prob,
                    layer_norm_before_residual=self.layer_norm_before_residual,
                )
            )

        layers.append(
            NoTorchLinear(
                weight=matrices[-1],
            )
        )

        return NoTorchSequential(layers, num_queries=batch_size)

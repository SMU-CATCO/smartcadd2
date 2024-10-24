from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
import haiku as hk
import jraph


from .jax_layers import GATLayer, EGNNLayer


class GAT(hk.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_layers,
        concat=True,
        name=None,
    ):
        super().__init__(name="GAT")
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.concat = concat

        self.init_fn = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        self.input_layer = lambda x: jax.nn.relu(hk.Linear(dim, w_init=self.init_fn)(x))

        self.gat_layers = [
            GATLayer(dim, num_heads, concat, name=f"gat_layer_{i}")
            for i in range(num_layers)
        ]

        self.skip_connections = [
            hk.Linear(dim, w_init=self.init_fn, name=f"skip_connection_{i}")
            for i in range(num_layers)
        ]

        self.mlp = hk.Sequential([
            hk.Linear(dim, w_init=self.init_fn),
            lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            hk.Linear(dim, w_init=self.init_fn),
            lambda x: jax.nn.leaky_relu(x, negative_slope=0.2),
            hk.Linear(1, w_init=self.init_fn),
        ])

    def __call__(self, graph):
        nodes, _, _, _, _, n_node, _ = graph
        nodes = nodes["x"]
        sum_n_node = jax.tree_util.tree_leaves(nodes)[0].shape[0]

        x = self.input_layer(nodes)

        for i in range(self.num_layers):
            graph_out = self.gat_layers[i](graph._replace(nodes=x))

            # skip connection
            x = x + self.skip_connections[i](graph_out.nodes)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)

        # global mean pooling
        graph_idx = jnp.repeat(jnp.arange(n_node.shape[0]), n_node, total_repeat_length=sum_n_node)
        x = jraph.segment_mean(x, segment_ids=graph_idx, num_segments=n_node.shape[0])

        # final mlp
        return self.mlp(x).reshape(-1)

# Adapted from https://github.com/gerkone/egnn-jax/blob/main/egnn_jax/egnn.py
class EGNN(hk.Module):
    r"""
    E(n) Graph Neural Network (https://arxiv.org/abs/2102.09844).

    Original implementation: https://github.com/vgsatorras/egnn
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        act_fn: Callable = jax.nn.silu,
        num_layers: int = 4,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
    ):
        r"""
        Initialize the network.

        Args:
            hidden_size: Number of hidden features
            output_size: Number of features for 'h' at the output
            act_fn: Non-linearity
            num_layers: Number of layer for the EGNN
            residual: Use residual connections, we recommend not changing this one
            attention: Whether using attention or not
            normalize: Normalizes the coordinates messages such that:
                x^{l+1}_i = x^{l}_i + \sum(x_i - x_j)\phi_x(m_{ij})\|x_i - x_j\|
                It may help in the stability or generalization. Not used in the paper.
            tanh: Sets a tanh activation function at the output of \phi_x(m_{ij}). It
                bounds the output of \phi_x(m_{ij}) which definitely improves in
                stability but it may decrease in accuracy. Not used in the paper.
        """
        super().__init__()

        self._hidden_size = hidden_size
        self._output_size = output_size
        self._act_fn = act_fn
        self._num_layers = num_layers
        self._residual = residual
        self._attention = attention
        self._normalize = normalize
        self._tanh = tanh

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply EGNN.

        Args:
            graph: Input graph
            pos: Node position
            edge_attribute: Edge attribute (optional)
            node_attribute: Node attribute (optional)

        Returns:
            Tuple of updated node features and positions
        """
        init_fn = hk.initializers.VarianceScaling(  
            scale=1.0, mode="fan_avg", distribution="uniform"
        )
        # input node embedding
        h = hk.Linear(self._hidden_size, w_init=init_fn, name="embedding")(graph.nodes)
        graph = graph._replace(nodes=h)

        # message passing
        for n in range(self._num_layers):
            graph, pos = EGNNLayer(
                layer_num=n,
                hidden_size=self._hidden_size,
                output_size=self._hidden_size,
                act_fn=self._act_fn,
                residual=self._residual,
                attention=self._attention,
                normalize=self._normalize,
                tanh=self._tanh,
            )(
                graph,
                pos,
                edge_attribute=edge_attribute,
                node_attribute=node_attribute,
            )
        # node readout
        h = hk.Linear(self._output_size, w_init=init_fn, name="readout")(graph.nodes)
        return h, pos


# # Adapted from: https://github.com/google-deepmind/jraph/blob/master/jraph/_src/models.py
# def GAT_Layer(
#     num_heads: int,
#     per_head_channels: int,
#     attention_query_fn: Callable,
#     attention_logit_fn: Callable,
#     node_update_fn: Optional[Callable] = None,
#     edge_update_fn: Optional[Callable] = None,
# ):
#     """Returns a method that applies a Graph Attention Network layer with multi-head attention.

#     This implementation follows the GATv2 approach, similar to GATv2Conv in PyTorch Geometric.

#     Args:
#       num_heads: Number of attention heads.
#       per_head_channels: Number of output channels per attention head.
#       attention_query_fn: function that generates linear transformations for each head.
#       attention_logit_fn: function that computes attention logits.
#       node_update_fn: function that updates the aggregated messages. If None,
#         will concatenate the multi-head outputs (or average if concat=False).
#       edge_update_fn: Optional function to update edge features (not used in this implementation).

#     Returns:
#       A function that applies a multi-head Graph Attention layer.
#     """
#     if node_update_fn is None:
#         node_update_fn = (
#             lambda x: x
#         )  # Identity function, as concatenation is done internally

#     def _ApplyGAT(graph):
#         """Applies a multi-head Graph Attention layer."""
#         nodes, edges, receivers, senders, _, _, _ = graph
#         sum_n_node = nodes.shape[0]

#         # Linear transformation for each head
#         nodes_transformed = attention_query_fn(nodes)
#         nodes_transformed = nodes_transformed.reshape(
#             -1, num_heads, per_head_channels
#         )

#         # Compute attention logits
#         sent_attributes = nodes_transformed[senders]
#         received_attributes = nodes_transformed[receivers]
#         attention_logits = attention_logit_fn(
#             sent_attributes, received_attributes
#         )

#         # Apply softmax to get attention coefficients
#         alpha = jraph.segment_softmax(
#             attention_logits, segment_ids=receivers, num_segments=sum_n_node
#         )

#         # Apply attention coefficients
#         out = sent_attributes * alpha

#         # Aggregate messages
#         out = jraph.segment_sum(
#             out, segment_ids=receivers, num_segments=sum_n_node
#         )

#         # Concatenate or average the multi-head results
#         out = out.reshape(sum_n_node, -1)  # Concatenate by default
#         # If you want to average instead, use:
#         # out = out.mean(axis=1)

#         # Apply final update function
#         nodes = node_update_fn(out)

#         return graph._replace(nodes=nodes)

#     return _ApplyGAT

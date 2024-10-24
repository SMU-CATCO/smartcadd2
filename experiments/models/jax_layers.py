from typing import Optional, Callable, Any, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import haiku as hk
import jraph


class GATLayer(hk.Module):
    def __init__(
        self,
        dim,
        num_heads,
        concat=True,
        name=None,
    ):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.per_head_channels = dim // num_heads
        self.concat = concat

        self.init_fn = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        # node update function
        self.node_update_fn = lambda x: x if concat else jnp.mean(x, axis=1)

        # attention functions
        self.attention_query_fn = lambda x: hk.Linear(
            dim, w_init=self.init_fn
        )(x)

        self.attention_logit_fn = lambda q, k: hk.Linear(
            1, w_init=self.init_fn
        )(
            jax.nn.leaky_relu(
                jnp.concatenate([q, k], axis=-1), negative_slope=0.2
            )
        )

    def __call__(self, graph):
        nodes, _, receivers, senders, _, _, _ = graph
        sum_n_node = nodes.shape[0]

        # Linear transformation for each head
        nodes_transformed = self.attention_query_fn(nodes)
        nodes_transformed = nodes_transformed.reshape(
            -1, self.num_heads, self.per_head_channels
        )

        # Compute attention logits
        sent_attributes = nodes_transformed[senders]
        received_attributes = nodes_transformed[receivers]
        attention_logits = self.attention_logit_fn(
            sent_attributes, received_attributes
        )

        # Apply softmax to get attention coefficients
        alpha = jraph.segment_softmax(
            attention_logits, segment_ids=receivers, num_segments=sum_n_node
        )

        # Apply attention coefficients
        out = sent_attributes * alpha

        # Aggregate messages
        out = jraph.segment_sum(
            out, segment_ids=receivers, num_segments=sum_n_node
        )

        # Concatenate or average the multi-head results
        out = out.reshape(sum_n_node, -1)  # Concatenate by default

        # Apply final update function
        nodes = self.node_update_fn(out)

        return graph._replace(nodes=nodes)

# Adapted from https://github.com/gerkone/egnn-jax/blob/main/egnn_jax/egnn.py
class EGNNLayer(hk.Module):
    """EGNN layer.

    Args:
        layer_num: layer number
        hidden_size: hidden size
        output_size: output size
        blocks: number of blocks in the node and edge MLPs
        act_fn: activation function
        pos_aggregate_fn: position aggregation function
        msg_aggregate_fn: message aggregation function
        residual: whether to use residual connections
        attention: whether to use attention
        normalize: whether to normalize the coordinates
        tanh: whether to use tanh in the position update
        dt: position update step size
        eps: small number to avoid division by zero
    """

    def __init__(
        self,
        layer_num: int,
        hidden_size: int,
        output_size: int,
        blocks: int = 1,
        act_fn: Callable = jax.nn.silu,
        pos_aggregate_fn: Optional[Callable] = jraph.segment_sum,
        msg_aggregate_fn: Optional[Callable] = jraph.segment_sum,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        dt: float = 0.001,
        eps: float = 1e-8,
    ):
        super().__init__(f"layer_{layer_num}")

        self.init_fn = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )

        # message network
        self._edge_mlp = hk.nets.MLP(
            [hidden_size] * blocks + [hidden_size],
            w_init=self.init_fn,
            activation=act_fn,
            activate_final=True,
        )

        # update network
        self._node_mlp = hk.nets.MLP(
            [hidden_size] * blocks + [output_size],
            w_init=self.init_fn,
            activation=act_fn,
            activate_final=False,
        )

        # position update network
        net = [hk.Linear(hidden_size, w_init=self.init_fn), act_fn]

        # NOTE: from https://github.com/vgsatorras/egnn/blob/main/models/gcl.py#L254
        net += [hk.Linear(1, with_bias=False, w_init=hk.initializers.UniformScaling(dt))]
        if tanh:
            net.append(jax.nn.tanh)
        self._pos_correction_mlp = hk.Sequential(net)

        # attention
        self._attention_mlp = None
        if attention:
            self._attention_mlp = hk.Sequential(
                [hk.Linear(hidden_size, w_init=self.init_fn), jax.nn.sigmoid]
            )

        self.pos_aggregate_fn = pos_aggregate_fn
        self.msg_aggregate_fn = msg_aggregate_fn
        self._residual = residual
        self._normalize = normalize
        self._eps = eps

    def _pos_update(
        self,
        pos: jnp.ndarray,
        graph: jraph.GraphsTuple,
        coord_diff: jnp.ndarray,
    ) -> jnp.ndarray:
        trans = coord_diff * self._pos_correction_mlp(graph.edges)
        # NOTE: was in the original code
        trans = jnp.clip(trans, -100, 100)
        return self.pos_aggregate_fn(
            trans, graph.senders, num_segments=pos.shape[0]
        )

    def _message(
        self,
        radial: jnp.ndarray,
        edge_attribute: jnp.ndarray,
        edge_features: Any,
        incoming: jnp.ndarray,
        outgoing: jnp.ndarray,
        globals_: Any,
    ) -> jnp.ndarray:
        _ = edge_features
        _ = globals_
        msg = jnp.concatenate([incoming, outgoing, radial], axis=-1)
        if edge_attribute is not None:
            msg = jnp.concatenate([msg, edge_attribute], axis=-1)
        msg = self._edge_mlp(msg)
        if self._attention_mlp:
            att = self._attention_mlp(msg)
            msg = msg * att
        return msg

    def _update(
        self,
        node_attribute: jnp.ndarray,
        nodes: jnp.ndarray,
        senders: Any,
        msg: jnp.ndarray,
        globals_: Any,
    ) -> jnp.ndarray:
        _ = senders
        _ = globals_
        x = jnp.concatenate([nodes, msg], axis=-1)
        if node_attribute is not None:
            x = jnp.concatenate([x, node_attribute], axis=-1)
        x = self._node_mlp(x)
        if self._residual:
            x = nodes + x
        return x

    def _coord2radial(
        self, graph: jraph.GraphsTuple, coord: jnp.array
    ) -> Tuple[jnp.array, jnp.array]:
        coord_diff = coord[graph.senders] - coord[graph.receivers]
        radial = jnp.sum(coord_diff**2, 1)[:, jnp.newaxis]
        if self._normalize:
            norm = jnp.sqrt(radial)
            coord_diff = coord_diff / (norm + self._eps)
        return radial, coord_diff

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        pos: jnp.ndarray,
        edge_attribute: Optional[jnp.ndarray] = None,
        node_attribute: Optional[jnp.ndarray] = None,
    ) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        """
        Apply EGNN layer.

        Args:
            graph: Graph from previous step
            pos: Node position, updated separately
            edge_attribute: Edge attribute (optional)
            node_attribute: Node attribute (optional)
        """
        radial, coord_diff = self._coord2radial(graph, pos)

        graph = jraph.GraphNetwork(
            update_edge_fn=Partial(self._message, radial, edge_attribute),
            update_node_fn=Partial(self._update, node_attribute),
            aggregate_edges_for_nodes_fn=self.msg_aggregate_fn,
        )(graph)

        pos = pos + self._pos_update(pos, graph, coord_diff)

        return graph, pos

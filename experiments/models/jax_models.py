from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk
import jraph
from jraph import GATAttentionQueryFn, GATAttentionLogitFn, GATNodeUpdateFn


# Adapted from: https://github.com/google-deepmind/jraph/blob/master/jraph/_src/models.py
def GAT_Layer(
    attention_query_fn: GATAttentionQueryFn,
    attention_logit_fn: GATAttentionLogitFn,
    node_update_fn: Optional[GATNodeUpdateFn] = None,
):
    """Returns a method that applies a Graph Attention Network layer.

    Graph Attention message passing as described in
    https://arxiv.org/abs/1710.10903. This model expects node features as a
    jnp.array, may use edge features for computing attention weights, and
    ignore global features. It does not support nests.

    NOTE: this implementation assumes that the input graph has self edges. To
    recover the behavior of the referenced paper, please add self edges.

    Args:
      attention_query_fn: function that generates attention queries
        from sender node features.
      attention_logit_fn: function that converts attention queries into logits for
        softmax attention.
      node_update_fn: function that updates the aggregated messages. If None,
        will apply leaky relu and concatenate (if using multi-head attention).

    Returns:
      A function that applies a Graph Attention layer.
    """
    # pylint: disable=g-long-lambda
    if node_update_fn is None:
        # By default, apply the leaky relu and then concatenate the heads on the
        # feature axis.
        node_update_fn = lambda x: jnp.reshape(
            jax.nn.leaky_relu(x), (x.shape[0], -1)
        )

    def _ApplyGAT(graph):
        """Applies a Graph Attention layer."""
        nodes, edges, receivers, senders, _, _, _ = graph
        # Equivalent to the sum of n_node, but statically known.
        try:
            sum_n_node = nodes.shape[0]
        except IndexError:
            raise IndexError(
                "GAT requires node features"
            )  # pylint: disable=raise-missing-from

        # First pass nodes through the node updater.
        nodes = attention_query_fn(nodes)
        # pylint: disable=g-long-lambda
        # We compute the softmax logits using a function that takes the
        # embedded sender and receiver attributes.
        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]
        softmax_logits = attention_logit_fn(
            sent_attributes, received_attributes, edges
        )

        # Compute the softmax weights on the entire tree.
        weights = jraph.segment_softmax(
            softmax_logits, segment_ids=receivers, num_segments=sum_n_node
        )
        # Apply weights
        messages = sent_attributes * weights
        # Aggregate messages to nodes.
        nodes = jraph.segment_sum(
            messages, segment_ids=receivers, num_segments=sum_n_node
        )

        # Apply an update function to the aggregated messages.
        nodes = node_update_fn(nodes)
        return graph._replace(nodes=nodes)

    return _ApplyGAT
 
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import haiku as hk
import jraph


# Adapted from: https://github.com/google-deepmind/jraph/blob/master/jraph/_src/models.py
def GAT_Layer(
    num_heads: int,
    per_head_channels: int,
    attention_query_fn: Callable,
    attention_logit_fn: Callable,
    node_update_fn: Optional[Callable] = None,
    edge_update_fn: Optional[Callable] = None,
):
    """Returns a method that applies a Graph Attention Network layer with multi-head attention.

    This implementation follows the GATv2 approach, similar to GATv2Conv in PyTorch Geometric.

    Args:
      num_heads: Number of attention heads.
      per_head_channels: Number of output channels per attention head.
      attention_query_fn: function that generates linear transformations for each head.
      attention_logit_fn: function that computes attention logits.
      node_update_fn: function that updates the aggregated messages. If None,
        will concatenate the multi-head outputs (or average if concat=False).
      edge_update_fn: Optional function to update edge features (not used in this implementation).

    Returns:
      A function that applies a multi-head Graph Attention layer.
    """
    if node_update_fn is None:
        node_update_fn = lambda x: x  # Identity function, as concatenation is done internally

    def _ApplyGAT(graph):
        """Applies a multi-head Graph Attention layer."""
        nodes, edges, receivers, senders, _, _, _ = graph
        sum_n_node = nodes.shape[0]

        # Linear transformation for each head
        nodes_transformed = attention_query_fn(nodes)
        nodes_transformed = nodes_transformed.reshape(-1, num_heads, per_head_channels)

        # Compute attention logits
        sent_attributes = nodes_transformed[senders]
        received_attributes = nodes_transformed[receivers]
        attention_logits = attention_logit_fn(sent_attributes, received_attributes)

        # Apply softmax to get attention coefficients
        alpha = jraph.segment_softmax(
            attention_logits, segment_ids=receivers, num_segments=sum_n_node
        )

        # Apply attention coefficients
        out = sent_attributes * alpha

        # Aggregate messages
        out = jraph.segment_sum(out, segment_ids=receivers, num_segments=sum_n_node)

        # Concatenate or average the multi-head results
        out = out.reshape(sum_n_node, -1)  # Concatenate by default
        # If you want to average instead, use:
        # out = out.mean(axis=1)

        # Apply final update function
        nodes = node_update_fn(out)

        return graph._replace(nodes=nodes)

    return _ApplyGAT


# def GAT_Layer(
#     num_heads: int,
#     per_head_channels: int,
#     attention_query_fn: callable,
#     attention_logit_fn: callable,
#     node_update_fn: callable = None,
#     edge_update_fn: callable = None,
#     use_edge_features: bool = False,
# ):
#     """Returns a method that applies a Graph Attention Network layer.

#     This implementation supports multi-head attention and edge features.

#     Args:
#       num_heads: Number of attention heads.
#       per_head_channels: Number of channels per attention head.
#       attention_query_fn: Function that generates attention queries from node features.
#       attention_key_fn: Function that generates attention keys from node features.
#       attention_value_fn: Function that generates attention values from node features.
#       attention_logit_fn: Function that converts attention queries, keys, and edge features into logits.
#       node_update_fn: Function that updates the aggregated messages. If None,
#         will apply leaky relu and concatenate the heads.

#     Returns:
#       A function that applies a Graph Attention layer.
#     """
#     if node_update_fn is None:

#         def node_update_fn(x):
#             x = jax.nn.leaky_relu(x, negative_slope=0.2)
#             return jnp.reshape(x, (x.shape[0], -1))

#     if use_edge_features:
#         # make sure attention_logit_fn takes edge features
#         assert (
#             attention_logit_fn.__code__.co_argcount == 3
#         ), "attention_logit_fn should take 3 arguments when use_edge_features is True"
#     else:
#         # make sure attention_logit_fn takes only queries and keys
#         assert (
#             attention_logit_fn.__code__.co_argcount == 2
#         ), "attention_logit_fn should take 2 arguments when use_edge_features is False"

#     def _ApplyGAT(graph):
#         """Applies a Graph Attention layer."""
#         nodes, edges, receivers, senders, _, _, _ = graph
#         sum_n_node = nodes.shape[0]

#         # Generate queries, keys, and values
#         nodes = attention_query_fn(nodes)

#         # Reshape for multi-head attention
#         queries = queries.reshape(-1, num_heads, per_head_channels)
#         keys = keys.reshape(-1, num_heads, per_head_channels)
#         values = values.reshape(-1, num_heads, per_head_channels)

#         # Compute attention logits
#         sent_attributes = keys[senders]
#         received_attributes = queries[receivers]

#         if use_edge_features:
#             edge_attributes = edges.reshape(edges.shape[0], 1, -1).repeat(
#                 num_heads, axis=1
#             )
#             attention_logits = attention_logit_fn(
#                 received_attributes, sent_attributes, edge_attributes
#             )
#         else:
#             attention_logits = attention_logit_fn(
#                 received_attributes, sent_attributes
#             )

#         # Compute attention weights
#         attention_weights = jraph.segment_softmax(
#             attention_logits, segment_ids=receivers, num_segments=sum_n_node
#         )

#         # Apply attention weights to values
#         messages = values[senders] * attention_weights

#         # Aggregate messages
#         aggregated_messages = jraph.segment_sum(
#             messages, segment_ids=receivers, num_segments=sum_n_node
#         )

#         # Update node features
#         new_nodes = node_update_fn(aggregated_messages)

#         # Update edge features if edge_update_fn is provided
#         if edge_update_fn is not None:
#             new_edges = edge_update_fn(edges, attention_weights)
#             return graph._replace(nodes=new_nodes, edges=new_edges)
#         else:
#             return graph._replace(nodes=new_nodes)

#     return _ApplyGAT

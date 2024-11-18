import torch
from torch_geometric.utils import remove_self_loops

def select_target(data, target):
    data = data.clone()
    data.y = data.y[:, target]
    return data


def make_complete(data):
    data = data.clone()
    device = data.edge_index.device

    row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
    col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

    row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
    col = col.repeat(data.num_nodes)
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = None
    if data.edge_attr is not None:
        idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
        size = list(data.edge_attr.size())
        size[0] = data.num_nodes * data.num_nodes
        edge_attr = data.edge_attr.new_zeros(size)
        edge_attr[idx] = data.edge_attr

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    data.edge_attr = edge_attr
    data.edge_index = edge_index

    return data

def make_radius_graph(data, cutoff):
    """Creates edge_index and edge_attr based on interatomic distance cutoff.
    
    Args:
        data (Data): PyG Data object containing node positions
        cutoff (float): Distance cutoff for connecting nodes
        
    Returns:
        Data: Updated PyG Data object with new edge_index and edge_attr
    """
    data = data.clone()
    device = data.pos.device
    num_nodes = data.pos.shape[0]

    # Create all possible edges
    row = torch.arange(num_nodes, dtype=torch.long, device=device)
    col = torch.arange(num_nodes, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
    col = col.repeat(num_nodes)
    
    # Calculate pairwise distances
    dist = torch.norm(data.pos[row] - data.pos[col], dim=1)
    
    # Keep edges within cutoff
    mask = (dist < cutoff) & (row != col)
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    
    # Update edge attributes with distances
    edge_attr = dist[mask].unsqueeze(-1)

    # remove self loops
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    

    return data

def add_pairwise_distances(data):
    data = data.clone()
    device = data.pos.device
    num_nodes = data.pos.shape[0]
    row = torch.arange(num_nodes, dtype=torch.long, device=device)
    col = torch.arange(num_nodes, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
    col = col.repeat(num_nodes)
    dist = torch.norm(data.pos[row] - data.pos[col], dim=1)

    # remove self loops
    _, edge_attr = remove_self_loops(torch.stack([row, col]), dist.unsqueeze(-1))

    data.edge_attr = edge_attr
    return data

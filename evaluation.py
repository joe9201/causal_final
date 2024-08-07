import networkx as nx

def calculate_shd(true_graph, estimated_graph):
    """Calculate the Structural Hamming Distance (SHD) between the true and estimated graphs."""
    true_edges = set(true_graph.edges())
    estimated_edges = set(estimated_graph.edges())
    
    extra_edges = estimated_edges - true_edges
    missing_edges = true_edges - estimated_edges
    
    shd = len(extra_edges) + len(missing_edges)
    return shd

def calculate_recall_precision(true_graph, estimated_graph):
    """Calculate the recall and precision of the estimated graph."""
    true_edges = set(true_graph.edges())
    estimated_edges = set(estimated_graph.edges())
    
    true_positives = len(true_edges & estimated_edges)
    false_positives = len(estimated_edges - true_edges)
    false_negatives = len(true_edges - estimated_edges)
    
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    return recall, precision

def evaluate_graph(estimated_graph, true_graph):
    """
    Evaluates the estimated causal graph against the true graph.
    
    Args:
        estimated_graph (networkx.DiGraph): The estimated causal graph.
        true_graph (networkx.DiGraph): The true causal graph.
        
    Returns:
        tuple: A tuple containing the following evaluation metrics:
            * shd (int): Structural Hamming Distance.
            * recall (float): True positive rate (recall).
            * precision (float): Precision of the estimated graph.
    """
    shd = calculate_shd(true_graph, estimated_graph)
    recall, precision = calculate_recall_precision(true_graph, estimated_graph)
    return shd, recall, precision
import numpy as np
import networkx as nx
import os
import sys
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils
from data_preparation import load_and_prepare_adult_data
from plotting_utils import plot_and_save_graph

def create_background_knowledge(labels):
    """
    Create background knowledge for the given labels to ensure certain nodes are root nodes.

    Args:
        labels (list): List of labels for the data columns.

    Returns:
        BackgroundKnowledge: The background knowledge object with specified rules.
    """
    root_nodes = {'race', 'age', 'sex', 'native.country'}
    bk = BackgroundKnowledge()
    
    # Convert labels to GraphNode instances
    nodes = {label: GraphNode(label) for label in labels}
    
    # Add root node constraints (no incoming edges)
    for root_node in root_nodes:
        if root_node in nodes:
            for node in nodes:
                if node != root_node:
                    bk.add_forbidden_by_node(nodes[node], nodes[root_node])

    return bk

def run_pc_algorithm(data, labels, output_dir):
    """
    Run the PC algorithm with background knowledge.

    Args:
        data (np.ndarray): The input data for causal discovery.
        labels (list): List of labels for the data columns.
        output_dir (str): Directory to save the output graph.

    Returns:
        None
    """
    try:
        # Create background knowledge
        bk = create_background_knowledge(labels)

        # Run PC algorithm
        print(f"Running PC with background knowledge")
        cg_pc = pc(data, alpha=0.05, indep_test=fisherz, stable=True, uc_rule=0, background_knowledge=bk)

        # Convert CausalLearn Graph to NetworkX graph
        nx_graph = nx.DiGraph()
        for edge in cg_pc.G.get_graph_edges():
            if edge.endpoint1 == '->' and edge.endpoint2 == '->':
                nx_graph.add_edge(edge.node1.name, edge.node2.name)

        # Plot and save the graph
        filename = 'pc_graph_with_background_knowledge.png'
        filepath = os.path.join(output_dir, filename)
        plot_and_save_graph(nx_graph, labels, filepath)

        print(f"Graph saved at: {filepath}")
    except Exception as e:
        print(f"Error in run_pc_algorithm: {str(e)}")
        raise

# Main function to load data and run PC algorithm
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pc_algorithm.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    if dataset == 'student':
        file_path = r'C:\Users\adams\OneDrive\Desktop\causal test\data\student-por_raw.csv'
        df_encoded, labels, data = load_and_prepare_student_data(file_path)
        output_dir = r'C:\Users\adams\OneDrive\Desktop\causal test\student_DAGS\pc_logic'
    elif dataset == 'adult':
        file_path = r'C:\Users\adams\OneDrive\Desktop\causal test\data\adult_cleaned.csv'
        df_encoded, labels, data = load_and_prepare_adult_data(file_path)
        output_dir = r'C:\Users\adams\OneDrive\Desktop\causal test\adult_DAGS\pc_logic'
    else:
        print(f"Invalid dataset argument: {dataset}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_pc_algorithm(data, labels, output_dir)


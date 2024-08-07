import os
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from plotting_utils import causal_learn_to_networkx, plot_and_save_graph

def run_pc_algorithm(data, labels, alpha=0.1, stable=False, uc_rule=2, output_dir='output'):
    """Runs the PC algorithm and returns the estimated causal graph."""
    # Convert boolean columns to integers
    data = data.astype(float)

    try:
        print(f"Running PC algorithm with alpha={alpha}, stable={stable}, uc_rule={uc_rule}")
        cg_pc = pc(data, alpha=alpha, stable=stable, uc_rule=uc_rule, indep_test=fisherz)

        # Convert CausalLearn GeneralGraph to NetworkX DiGraph using the utility function
        nx_graph = causal_learn_to_networkx(cg_pc.G)

        # Ensure the graph is a DAG by removing bidirectional edges
        bidirectional_edges = [(u, v) for u, v in nx_graph.edges if nx_graph.has_edge(v, u)]
        for u, v in bidirectional_edges:
            nx_graph.remove_edge(v, u)

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define the full path for the output file
        filename = os.path.join(output_dir, 'pc_graph.png')

        # Use the updated plot_and_save_graph function to save the graph
        plot_and_save_graph(nx_graph, labels, filename)

        print(f"Graph saved at: {filename}")
        return nx_graph
    except Exception as e:
        print(f"Error running PC algorithm: {e}")
        raise
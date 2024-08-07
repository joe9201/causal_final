import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx
from causallearn.graph.GeneralGraph import GeneralGraph

def plot_and_save_graph(graph, labels, filename):
    """
    Plots a causal graph using pydot and saves it to a file.

    Args:
        graph (nx.DiGraph): The NetworkX graph to plot.
        labels (list): A list of labels for the nodes.
        filename (str): The name of the file to save the plot.
    """
    # Create a copy of the graph with integer node labels
    graph_copy = nx.DiGraph(graph)

    # Set labels as attributes for pydot
    for i, node in enumerate(graph_copy.nodes()):
        graph_copy.nodes[node]['label'] = labels[i]

    pyd = to_pydot(graph_copy)

    # Adjust node font size and edge thickness
    for node in pyd.get_nodes():
        node.set_fontsize(12)
    for edge in pyd.get_edges():
        edge.set_penwidth(2)

    # Convert to DOT string for debugging
    dot_str = pyd.to_string()
    with open(f"{filename}.dot", "w") as f:
        f.write(dot_str)

    # Check for syntax issues
    print(f"DOT file content for debugging:\n{dot_str}")

    # Plot and save the graph with higher resolution
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    try:
        pyd.write_png(filename, prog='dot')  # Use 'dot' for layout and higher DPI
    except Exception as e:
        print(f"Error saving graph with pydot: {e}")

    # Display the graph with pause
    img = mpimg.imread(filename)
    plt.axis('off')
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(0.001)  # Small pause for some backends
    plt.close()  # Close the plot window

def causal_learn_to_networkx(causal_learn_graph):
    """
    Convert a CausalLearn GeneralGraph to a NetworkX DiGraph.

    Args:
        causal_learn_graph (GeneralGraph): The CausalLearn graph to convert.

    Returns:
        nx.DiGraph: The converted NetworkX directed graph.
    """
    nx_graph = nx.DiGraph()

    for edge in causal_learn_graph.get_graph_edges():
        nx_graph.add_edge(edge.node1, edge.node2)

    return nx_graph

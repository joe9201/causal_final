from causallearn.search.FCMBased import lingam
import networkx as nx
from plotting_utils import plot_and_save_graph

def run_lingam_algorithm(data, labels):
    model_lingam = lingam.ICALiNGAM(max_iter=500)
    model_lingam.fit(data)
    
    adjacency_matrix = model_lingam.adjacency_matrix_

    # Create NetworkX graph for evaluation
    lingam_graph = nx.DiGraph(adjacency_matrix)

    # Relabel nodes using the provided labels
    mapping = {i: labels[i] for i in range(len(labels))}
    lingam_graph = nx.relabel_nodes(lingam_graph, mapping)

    plot_and_save_graph(lingam_graph, labels, 'lingam_graph.png')

    return lingam_graph

if __name__ == "__main__":
    from data_preparation import load_and_prepare_student_data
    
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_student_data(file_path)
    lingam_graph = run_lingam_algorithm(data, labels)
    print("LiNGAM Algorithm graph created.")

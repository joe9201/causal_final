import argparse
import os
import traceback
import numpy as np
import networkx as nx
from lingam.direct_lingam import DirectLiNGAM
from plotting_utils import plot_and_save_graph
from data_preparation import load_and_prepare_student_data, load_and_prepare_adult_data
from true_graph import create_true_graph_student, create_true_graph_adult
from evaluation import evaluate_graph

def run_direct_lingam(data, labels, measure=None, output_dir='output'):
    """
    Run the DirectLiNGAM algorithm with specific parameters.

    Parameters:
    data (pd.DataFrame): The input data for causal discovery.
    labels (list): List of labels for the data columns.
    measure (str, optional): Measure to evaluate independence (None for default, 'pwling', 'pwling_fast').

    Returns:
    graph: The adjacency matrix representing the causal graph.
    """
    try:
        if measure:
            print(f"Running DirectLiNGAM with measure={measure}")
            model = DirectLiNGAM(measure=measure)
        else:
            print(f"Running DirectLiNGAM with default settings")
            model = DirectLiNGAM()
        
        model.fit(data)
        adjacency_matrix = model.adjacency_matrix_

        # Create NetworkX graph for evaluation
        direct_lingam_graph = nx.DiGraph(adjacency_matrix)

        # Relabel nodes using the provided labels
        mapping = {i: labels[i].replace('.', '_') for i in range(len(labels))}
        direct_lingam_graph = nx.relabel_nodes(direct_lingam_graph, mapping)

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define the full path for the output file
        filename = os.path.join(output_dir, 'direct_lingam_graph.png')

        # Use the consistent plot_and_save_graph function to save the graph
        plot_and_save_graph(direct_lingam_graph, labels, filename)

        print(f"Graph saved at: {filename}")
        return direct_lingam_graph
    except Exception as e:
        print(f"Error in run_direct_lingam: {str(e)}")
        raise

def main(dataset):
    if dataset == 'student':
        file_path = 'data/student-por_raw.csv'
        df_encoded, labels, data = load_and_prepare_student_data(file_path)
        true_graph = create_true_graph_student()
    elif dataset == 'adult':
        file_path = 'data/adult_cleaned.csv'  # Updated to use the cleaned adult data
        df_encoded, labels, data = load_and_prepare_adult_data(file_path)
        true_graph = create_true_graph_adult()
    else:
        raise ValueError("Invalid dataset. Choose either 'student' or 'adult'.")

    print(f"{dataset.capitalize()} Data Preparation:")
    print(df_encoded.dtypes)
    print("Missing values:\n", df_encoded.isnull().sum())

    graph = run_direct_lingam(data, labels)
    shd, recall, precision = evaluate_graph(graph, true_graph)
    print(f"SHD: {shd}, Recall: {recall}, Precision: {precision}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DirectLiNGAM Algorithm on a specified dataset.')
    parser.add_argument('--dataset', required=True, choices=['student', 'adult'], help='Dataset to use (student or adult)')
    args = parser.parse_args()

    main(args.dataset)

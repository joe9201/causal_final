import numpy as np
import traceback
import networkx as nx
from evaluation import evaluate_graph
from sklearn.model_selection import KFold
import argparse
import os
from plotting_utils import plot_and_save_graph
from direct_lingam import run_direct_lingam  # Importing the function from direct_lingam.py

def run_direct_lingam_default(data, labels, output_dir='output'):
    """
    Run the DirectLiNGAM algorithm with default parameters.

    Parameters:
    data (pd.DataFrame): The input data for causal discovery.
    labels (list): List of labels for the data columns.

    Returns:
    graph: The adjacency matrix representing the causal graph.
    """
    return run_direct_lingam(data, labels, measure=None, output_dir=output_dir)

def grid_search_direct_lingam(data, labels, true_graph, n_splits=5, output_dir="output"):
    param_grid = {
        'measure': ['default', 'pwling', 'pwling_fast']
    }

    best_params = None
    best_score = float('inf')

    # Set a consistent random seed for reproducibility
    np.random.seed(42)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for measure in param_grid['measure']:
        cv_scores = []

        for train_index, test_index in kf.split(data):
            train_data, test_data = data[train_index], data[test_index]

            try:
                if measure == 'default':
                    nx_graph = run_direct_lingam_default(train_data, labels, output_dir=output_dir)
                else:
                    nx_graph = run_direct_lingam(train_data, labels, measure=measure, output_dir=output_dir)
                shd, recall, precision = evaluate_graph(nx_graph, true_graph)
                cv_scores.append(shd)  # Using SHD as the score metric

            except Exception as e:
                print(f"Error with params: measure={measure} - {str(e)}")
                traceback.print_exc()  # Print the full traceback for detailed debugging

        avg_score = np.mean(cv_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = {'measure': measure}

        print(f"Params: measure={measure} - Avg SHD: {avg_score}")

    print(f"Best parameters for DirectLiNGAM Algorithm: {best_params}")
    print(f"Best score: {best_score}")

# Main function to load data and perform grid search
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune DirectLiNGAM Algorithm")
    parser.add_argument('--dataset', choices=['student', 'adult'], required=True, help='Dataset to use (student or adult)')
    args = parser.parse_args()

    if args.dataset == 'student':
        from data_preparation import load_and_prepare_student_data
        from true_graph import create_true_graph_student
        
        file_path = 'data/student-por_raw.csv'
        df_encoded, labels, data = load_and_prepare_student_data(file_path)
        true_graph = create_true_graph_student()
        output_dir = 'output/student_DAGS'
    elif args.dataset == 'adult':
        from data_preparation import load_and_prepare_adult_data
        from true_graph import create_true_graph_adult
        
        file_path = 'data/adult_cleaned.csv'
        df_encoded, labels, data = load_and_prepare_adult_data(file_path)
        true_graph = create_true_graph_adult()
        output_dir = 'output/adult_DAGS'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    grid_search_direct_lingam(data, labels, true_graph, output_dir=output_dir)

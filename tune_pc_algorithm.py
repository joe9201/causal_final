import numpy as np
import networkx as nx
import os
import sys
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from data_preparation import load_and_prepare_student_data, load_and_prepare_adult_data
from true_graph import create_true_graph_student, create_true_graph_adult
from evaluation import evaluate_graph
from plotting_utils import plot_and_save_graph, causal_learn_to_networkx
import traceback

# Function to run the PC algorithm with specific parameters
def run_pc_with_params(data, labels, alpha, stable, uc_rule, output_dir):
    try:
        print(f"Running PC with alpha={alpha}, stable={stable}, uc_rule={uc_rule}")
        cg_pc = pc(data, alpha=alpha, indep_test=fisherz, stable=stable, uc_rule=uc_rule)

        # Convert CausalLearn Graph to NetworkX graph
        nx_graph = causal_learn_to_networkx(cg_pc.G)

        # Plot and save the graph
        filename = f'pc_graph_alpha_{alpha}_stable_{stable}_uc_rule_{uc_rule}.png'
        filepath = os.path.join(output_dir, filename)
        plot_and_save_graph(nx_graph, labels, filepath)

        return nx_graph
    except Exception as e:
        print(f"Error in run_pc_with_params: alpha={alpha}, stable={stable}, uc_rule={uc_rule} - {str(e)}")
        traceback.print_exc()  # Print the full traceback for detailed debugging
        raise

# Function to perform grid search for PC algorithm
def grid_search_pc(data, labels, true_graph, output_dir):
    param_grid = {
        'alpha': [0.01, 0.05, 0.1],
        'stable': [True, False],
        'uc_rule': [0, 1, 2]
    }

    best_params = None
    best_score = float('inf')

    for alpha in param_grid['alpha']:
        for stable in param_grid['stable']:
            for uc_rule in param_grid['uc_rule']:
                try:
                    nx_graph = run_pc_with_params(data, labels, alpha, stable, uc_rule, output_dir)
                    shd, recall, precision = evaluate_graph(nx_graph, true_graph)
                    score = shd  # Using SHD as the score metric

                    if score < best_score:
                        best_score = score
                        best_params = {'alpha': alpha, 'stable': stable, 'uc_rule': uc_rule}

                    print(f"Params: alpha={alpha}, stable={stable}, uc_rule={uc_rule} - SHD: {shd}, Recall: {recall}, Precision: {precision}")
                except Exception as e:
                    print(f"Error with params: alpha={alpha}, stable={stable}, uc_rule={uc_rule} - {str(e)}")
                    traceback.print_exc()  # Print the full traceback for detailed debugging

    print(f"Best parameters for PC Algorithm: {best_params}")
    print(f"Best score: {best_score}")

# Main function to load data and perform grid search
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tune_pc_algorithm.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    if dataset == 'student':
        file_path = r'C:\Users\adams\OneDrive\Desktop\causal test\data\student-por_raw.csv'
        df_encoded, labels, data = load_and_prepare_student_data(file_path)
        true_graph = create_true_graph_student()
        output_dir = r'C:\Users\adams\OneDrive\Desktop\causal test\student_DAGS'
    elif dataset == 'adult':
        file_path = r'C:\Users\adams\OneDrive\Desktop\causal test\data\adult_cleaned.csv'
        df_encoded, labels, data = load_and_prepare_adult_data(file_path)
        true_graph = create_true_graph_adult()
        output_dir = r'C:\Users\adams\OneDrive\Desktop\causal test\adult_DAGS'
    else:
        print(f"Invalid dataset argument: {dataset}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    grid_search_pc(data, labels, true_graph, output_dir)

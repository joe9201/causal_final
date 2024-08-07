import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import argparse
from data_preparation import load_and_prepare_student_data, load_and_prepare_adult_data
from pc_algorithm import run_pc_algorithm
from lingam_algorithm import run_lingam_algorithm
from direct_lingam import run_direct_lingam
from true_graph import create_true_graph_student, create_true_graph_adult
from evaluation import evaluate_graph

def run_algorithms_for_dataset(data_preparation_func, true_graph_func, file_path, dataset_name, measure):
    df_encoded, labels, data = data_preparation_func(file_path)
    print(f"{dataset_name.capitalize()} Data Preparation:")
    print(df_encoded.dtypes)
    print("Missing values:\n", df_encoded.isnull().sum())
    print()

    graphs = {}  # Dictionary to store the graphs generated by each algorithm

    # Run Algorithms
    algorithms = {
        "PC": run_pc_algorithm,
        "LiNGAM": run_lingam_algorithm,
        "DirectLiNGAM": run_direct_lingam
    }

    for algo_name, algo_func in algorithms.items():
        print(f"\nRunning {algo_name} algorithm...")
        try:
            if algo_name == "DirectLiNGAM":
                graph = algo_func(data, labels, measure=measure)  # Pass the measure parameter
            else:
                graph = algo_func(data, labels)
            if graph is not None:  # Check if graph creation was successful
                graphs[algo_name] = graph
        except Exception as e:
            print(f"Error running {algo_name} algorithm: {e}")
        input("Press Enter to continue...")  # Pause until Enter is pressed

    if true_graph_func:
        G_true = true_graph_func()
        for algo_name, graph in graphs.items():
            if graph is not None:
                shd, recall, precision = evaluate_graph(graph, G_true)
                print(f"{algo_name} Algorithm - SHD: {shd}, Recall: {recall}, Precision: {precision}")
    else:
        print("True graph function not available for this dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Causal Discovery Algorithms on a specified dataset.')
    parser.add_argument('--dataset', required=True, choices=['student', 'adult'], help='Dataset to use (student or adult)')
    parser.add_argument('--measure', required=False, default='pwling', help='Measure to use for DirectLiNGAM (pwling, kernel, pwling_fast)')
    args = parser.parse_args()

    datasets = {
        'student': {
            "file_path": r'C:\Users\adams\OneDrive\Desktop\causal test\data\student-por_raw.csv',
            "data_preparation_func": load_and_prepare_student_data,
            "true_graph_func": create_true_graph_student
        },
        'adult': {
            "file_path": r'C:\Users\adams\OneDrive\Desktop\causal test\data\adult_cleaned.csv',
            "data_preparation_func": load_and_prepare_adult_data,
            "true_graph_func": create_true_graph_adult
        }
    }

    # Get the configuration for the selected dataset
    dataset_config = datasets.get(args.dataset)
    if dataset_config:
        run_algorithms_for_dataset(
            dataset_config["data_preparation_func"],
            dataset_config["true_graph_func"],
            dataset_config["file_path"],
            args.dataset,
            args.measure  # Pass the measure argument
        )
    else:
        print(f"Invalid dataset argument: {args.dataset}")

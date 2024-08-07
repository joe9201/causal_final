import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from causallib.estimation import IPW
from causalvis import CohortEvaluator, TreatmentEffectExplorer
from data_preparation import load_and_prepare_student_data
from matching import perform_matching
from dag_utils import load_dag  # Import the dag_utils module

def convert_types(data):
    """ Convert numpy data types to native Python types for JSON serialization. """
    if isinstance(data, dict):
        return {k: convert_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_types(i) for i in data]
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    else:
        return data

def main(data_file, dag_file):
    # Load and prepare data
    df_encoded, labels, data = load_and_prepare_student_data(data_file)
    
    # Load DAG and extract confounds and prognostics
    G, confounds, prognostics = load_dag(dag_file)
    
    # Check confounds and prognostics
    print("Confounds: ", confounds)
    print("Prognostics: ", prognostics)
    
    treatment = 'absences'
    outcome = 'G_avg'

    # Perform matching
    adjustedCohort, unadjustedCohort = perform_matching(df_encoded, confounds, prognostics, treatment, outcome)

    # Convert data types
    adjustedCohort = convert_types(adjustedCohort)
    unadjustedCohort = convert_types(unadjustedCohort)

    # Generate file names based on the DAG file name
    base_name = os.path.splitext(os.path.basename(dag_file))[0]
    adjusted_file_name = f'adjustedCohort_{base_name}.json'
    unadjusted_file_name = f'unadjustedCohort_{base_name}.json'

    # Save cohorts for visualization
    with open(adjusted_file_name, 'w') as f:
        json.dump(adjustedCohort, f, indent=4)
    with open(unadjusted_file_name, 'w') as f:
        json.dump(unadjustedCohort, f, indent=4)

    # Instantiate and visualize using CohortEvaluator
    cohort_evaluator = CohortEvaluator(unadjustedCohort=unadjustedCohort)
    display(cohort_evaluator)

    # Print the first three items of the selection and inverse selection
    print(cohort_evaluator.selection["confounds"][:3])
    print(cohort_evaluator.iselection["confounds"][:3])

    # Prepare data for TreatmentEffectExplorer
    for instance in adjustedCohort:
        instance['effect'] = instance['treatment'] - np.mean([x['outcome'] for x in adjustedCohort])

    # Instantiate and visualize using TreatmentEffectExplorer
    te_explorer = TreatmentEffectExplorer(data=adjustedCohort)
    display(te_explorer)

if __name__ == "__main__":
    data_file = 'data/student-por_raw.csv'
    dag_file = 'student_true_confounds.json'
    main(data_file, dag_file)
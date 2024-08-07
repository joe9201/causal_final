import networkx as nx
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import io
from networkx.drawing.nx_pydot import to_pydot
import matplotlib.image as mpimg

def create_true_graph_student():
    G_true = nx.DiGraph()

    pos = {
        "G_avg": (0.207, -0.688),
        "Medu": (-0.641, 0.589),
        "Pstatus": (-1.699, -0.614),
        "absences": (-1.531, 0.937),
        "failures": (0.873, 1.702),
        "famrel": (-2.270, -0.747),
        "famsup": (-1.316, -0.265),
        "health": (-0.945, 0.076),
        "higher": (-0.206, 0.589), 
        "internet": (0.302, 1.034),
        "paid": (-0.243, -1.400),
        "schoolsup": (0.867, -0.176),
        "studytime": (0.799, -1.289)
    }

    for node, position in pos.items():
        G_true.add_node(node, pos=position)

    G_true.add_edges_from([
        ("Medu", "G_avg"),
        ("Medu", "absences"),
        ("Medu", "higher"),
        ("Pstatus", "G_avg"),
        ("Pstatus", "absences"),
        ("Pstatus", "famrel"),
        ("failures", "G_avg"),
        ("failures", "absences"),
        ("famsup", "G_avg"),
        ("famsup", "absences"),
        ("health", "G_avg"),
        ("health", "absences"),
        ("higher", "G_avg"),
        ("internet", "G_avg"),
        ("internet", "absences"),
        ("paid", "G_avg"),
        ("schoolsup", "G_avg"),
        ("studytime", "G_avg")
    ])

    return G_true

def create_true_graph_adult():
    G_true = nx.DiGraph()
    
    # Define the positions of the nodes (optional, for visualization purposes)
    pos = {
        "age": (0, 0),
        "workclass": (1, 0),
        "education": (2, 0),
        "marital.status": (3, 0),
        "occupation": (4, 0),
        "relationship": (5, 0),
        "race": (6, 0),
        "sex": (7, 0),
        "hours.per.week": (8, 0),
        "native.country": (9, 0),
        "income": (10, 0)
    }

    for node, position in pos.items():
        G_true.add_node(node, pos=position)

    G_true.add_edges_from([
        ("age", "workclass"),
        ("education", "occupation"),
        ("marital.status", "occupation"),
        ("relationship", "income"),
        ("race", "income"),
        ("sex", "income"),
        ("hours.per.week", "income"),
        ("native.country", "income"),
        ("education", "income"),
        ("workclass", "income"),
        ("occupation", "income")
    ])

    return G_true

def plot_true_graph(G_true, dataset_name):
    pos = nx.get_node_attributes(G_true, 'pos')

    plt.figure(figsize=(12, 8))
    nx.draw(G_true, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True, arrowstyle='-|>', arrowsize=20)
    plt.title(f"True Causal Graph for {dataset_name.capitalize()} Dataset")
    plt.savefig(f'{dataset_name}_true_graph.png')
    plt.show()

if __name__ == "__main__":
    student_true_graph = create_true_graph_student()
    plot_true_graph(student_true_graph, "student")

    adult_true_graph = create_true_graph_adult()
    plot_true_graph(adult_true_graph, "adult")

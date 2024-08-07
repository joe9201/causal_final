import json
import networkx as nx

def load_dag(dag_file):
    with open(dag_file, 'r') as f:
        dag_data = json.load(f)
    G = nx.DiGraph()
    for node in dag_data['nodes']:
        G.add_node(node['name'])
    for link in dag_data['links']:
        G.add_edge(link['source']['name'], link['target']['name'])
    
    # Directly extract confounds and prognostics
    confounds = dag_data.get('confounds', [])
    prognostics = dag_data.get('prognostics', [])
    
    return G, confounds, prognostics
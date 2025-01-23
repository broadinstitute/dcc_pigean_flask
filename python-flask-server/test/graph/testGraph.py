

import networkx as nx
import json
from pyvis.network import Network

# Create a graph
G = nx.karate_club_graph()

# Convert to JSON
data = nx.node_link_data(G)
with open('./testGraph.json', 'w') as f:
    json.dump(data, f)

# convert to cytoscape data
data = nx.cytoscape_data(G)
json_data = json.dumps(data['elements'])  # This holds nodes and edges as needed by Cytoscape.js

# Optionally, write to a file
with open('testGraph.json', 'w') as f:
    f.write(json_data)

# export as pyvis
# net = Network()
# net.from_nx(G)
# net.show("graph.html")

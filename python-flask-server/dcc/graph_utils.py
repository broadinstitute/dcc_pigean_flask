
# imports
import numpy as np
import networkx as nx
import json
from pyvis.network import Network
import dcc.dcc_utils as dutils 
import dcc.data_utils as dautils 


# constants
logger = dutils.get_logger(__name__)


# methods
def generate_distinct_colors_orig(N):

    if N == 2:
        colors = [(1,0,0), (0,0,1)]
    else:

        colors = []
        step = 256 / N  # Step to space colors evenly in the RGB space

        white_scale = 0.5

        for i in range(1,N+1):
            r = 1 - white_scale * (1 - int((i * step) % 256) / 256.0)        # Red channel
            g = 1 - white_scale * (1 - int((i * step * 2) % 256) / 256.0)    # Green channel
            b = 1 - white_scale * (1 - int((i * step * 3) % 256) / 256.0)    # Blue channel
            colors.append((r, g, b))

    return colors


def generate_distinct_colors(N, start_with_red_blue=True):
    """
    Generate N distinct colors, ensuring complementarity if starting with red and blue.
    Defaults to the original behavior if not starting with red and blue.

    Args:
        N (int): The number of distinct colors to generate.
        start_with_red_blue (bool): If True, the first two colors are red and blue.

    Returns:
        list: A list of tuples representing RGB colors.
    """

    if N <= 0:
        return []

    colors = []

    if start_with_red_blue and N >= 2:
        # Start with predefined red and blue
        colors.append((1, 0, 0))  # Red
        colors.append((0, 0, 1))  # Blue

        import colorsys

        # Generate the remaining colors with complementarity
        for i in range(2, N):
            max_dist_color = None
            max_dist = -1

            # Sample potential colors in HSV space and find the most distinct one
            for h in range(0, 360, 10):  # Test hues in 10-degree increments
                for s in [0.7, 1.0]:     # Test two saturation levels
                    for v in [0.7, 1.0]: # Test two brightness levels
                        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
                        candidate_color = (r, g, b)

                        # Compute the minimum Euclidean distance to all existing colors
                        min_dist = min(
                            sum((r1 - r2)**2 for r1, r2 in zip(candidate_color, c))**0.5
                            for c in colors
                        )

                        # Keep track of the most distinct candidate
                        if min_dist > max_dist:
                            max_dist = min_dist
                            max_dist_color = candidate_color

            # Add the most distinct color to the list
            colors.append(max_dist_color)

    else:
        # Default behavior: evenly distribute colors in RGB space
        step = 256 / N  # Step to space colors evenly
        white_scale = 0.5

        for i in range(1, N + 1):
            r = 1 - white_scale * (1 - int((i * step) % 256) / 256.0)  # Red channel
            g = 1 - white_scale * (1 - int((i * step * 2) % 256) / 256.0)  # Green channel
            b = 1 - white_scale * (1 - int((i * step * 3) % 256) / 256.0)  # Blue channel
            colors.append((r, g, b))

    return colors

def normalized_to_rgb_list(list_color):
    # return tuple(int(255 * channel) for channel in color)
    # return tuple(int(255 * channel) for channel in color)
    list_result = []

    # loop through the colors
    for color in list_color:
        list_result.append(tuple(int(255 * channel) for channel in color))

    # return
    return list_result


def blend_colors(color_list, weights, opacity=1):
    '''
    will blend the colors provided
    '''
    weights = weights / np.sum(weights)

    if opacity < 0.1:
        opacity = 0.1
    if opacity > 1:
        opacity = 1


    color_list = np.array(color_list)

    blended_color = 1 - opacity * np.average(1 - color_list, axis=0, weights=weights)

    #blended_color = np.average(color_list, axis=0, weights=weights)



    blended_color[blended_color < 1/256.0] = 0
    blended_color[blended_color > 1] = 1


    return tuple(c for c in blended_color)  # Convert to 0-255 for HTML colors


def blend_rgb_colors(colors, weights, log=False):
    # Check if the number of colors and weights are equal
    if len(colors) != len(weights):
        raise ValueError("Each color must have a corresponding weight")

    # log
    if log:
        logger.info("got colors list: {}".format(colors))

    # Initialize sums and total weight
    total_weight = sum(weights)
    sum_r = sum_g = sum_b = 0

    # Iterate through each color and weight
    for color, weight in zip(colors, weights):
        # Parse the RGB values from the color string
        rgb = color.strip('rgb(').rstrip(')').split(',')
        r, g, b = map(int, rgb)
        
        # Add weighted RGB values
        sum_r += r * weight
        sum_g += g * weight
        sum_b += b * weight

    # Compute the weighted average of each component
    average_r = int(round(sum_r / total_weight))
    average_g = int(round(sum_g / total_weight))
    average_b = int(round(sum_b / total_weight))

    # Return the blended color in RGB format
    return f'rgb({average_r}, {average_g}, {average_b})'


def build_factor_graph_for_gui(list_factor, list_factor_genes, list_factor_gene_sets, test=False, log=True):
    '''
    will build the nodes and edges list for graph display

    - color of factor is generated
    - shape of factor is square
    ? size of factor (sum of factors) 
    - color of gene and gene sets is a weighted blending of the factors it is linked to
    '''
    graph = None
    map_nodes = {}

    # if test/debug
    if test:
        graph = build_test_graph(log=log)

    else:
        # get the extracted factors
        list_factor_graph = dautils.extract_factor_data_list(list_factor_input=list_factor, list_factor_genes_input=list_factor_genes, list_factor_gene_sets_input=list_factor_gene_sets)

        # log
        if log:
            logger.info("got list factor list of size: {}".format(len(list_factor_graph)))
            logger.info("got list factor list: {}".format(list_factor_graph))

        # start a new graph
        graph = nx.Graph()

        # get the colors needed for the number of factors
        num_colors = len(list_factor_graph)
        if log:
            logger.info("getting color list for factors of size: {}".format(num_colors))
        list_colors = generate_distinct_colors_orig(N=num_colors)

        # make rgb colors
        list_colors = normalized_to_rgb_list(list_color=list_colors)

        # add the factors as squares
        for index, factor in enumerate(list_factor_graph):
            color_node = "rgb{}".format(list_colors[index])
            label = factor.get('label')
            size = factor.get('gene_set_score', 1) * 10
            id_factor = factor.get('factor')
            if log:
                logger.info("adding factor to graph with index: {}, label: {} and color: {}".format(index, label, color_node))
            # will use string as key
            # graph.add_node("factor-{}".factor.get('label'), size=size, color=node_color, border_color=node_border_color, alpha=gene_node_opacity, label=node, gene=True)
            graph.add_node("factor-{}".format(id_factor), color=color_node, border_color=color_node, label=id_factor, size=size, shape='square')

        # get the extracted genes
        map_genes = dautils.extract_pigean_gene_factor_results_map(list_factor=list_factor, list_factor_genes=list_factor_genes, max_num_per_factor=20)

        for gene, list_value in map_genes.items():
            id_node_gene = "gene-{}".format(gene)

            # calculate the color
            color_gene = blend_rgb_colors(colors=[graph.nodes["factor-{}".format(row.get('factor'))].get('color') for row in list_value], weights=[row.get('factor_score') for row in list_value])
            size_gene = sum([row.get('factor_score') for row in list_value]) * 10

            # add the gene node
            graph.add_node(id_node_gene, label=gene, shape='circle', color=color_gene, size=size_gene)

            # add edge
            for factor_row in list_value:
                id_node_factor = factor_row.get('factor')
                score_edge = graph.nodes[id_node_factor].get('score')
                props_edge = {'score': score_edge, 'relatioship': 'has gene'}
                graph.add_edge(id_node_factor, id_node_gene, **props_edge)

        # get the extracted gene sets
        map_gene_sets = dautils.extract_pigean_gene_set_factor_results_map(list_factor=list_factor, list_factor_gene_sets=list_factor_gene_sets, max_num_per_factor=10)

        for gene_set, list_value in map_gene_sets.items():
            id_node_gene_set = "gene_set-{}".format(gene_set)

            # calculate the color
            color_gene_set = blend_rgb_colors(colors=[graph.nodes["factor-{}".format(row.get('factor'))].get('color') for row in list_value], weights=[row.get('factor_score') for row in list_value])
            size_gene_set = sum([row.get('factor_score') for row in list_value]) * 10

            # add the gene node
            graph.add_node(id_node_gene_set, label=gene_set, shape='diamond', color=color_gene_set, size=size_gene_set)

            # add edge
            for factor_row in list_value:
                id_node_factor = factor_row.get('factor')
                score_edge = graph.nodes[id_node_factor].get('score')
                props_edge = {'score': score_edge, 'relatioship': 'has gene set'}
                graph.add_edge(id_node_factor, id_node_gene_set, **props_edge)



    # return
    return graph


def build_factor_graph_for_llm(list_factor, list_factor_genes, list_factor_gene_sets, max_num_genes=20, max_num_gene_sets=10, test=False, log=True):
    '''
    will build the nodes and edges list for graph display

    - color of factor is generated
    - shape of factor is square
    ? size of factor (sum of factors) 
    - color of gene and gene sets is a weighted blending of the factors it is linked to
    '''
    graph = None
    map_nodes = {}

    # if test/debug
    if test:
        graph = build_test_graph(log=log)

    else:
        # get the extracted factors
        list_factor_graph = dautils.extract_factor_data_list(list_factor_input=list_factor, list_factor_genes_input=list_factor_genes, list_factor_gene_sets_input=list_factor_gene_sets)

        # log
        if log:
            logger.info("got list factor list of size: {}".format(len(list_factor_graph)))
            logger.info("got list factor list: {}".format(list_factor_graph))

        # start a new graph
        graph = nx.Graph()

        # add the factors as squares
        for index, factor in enumerate(list_factor_graph):
            label = factor.get('label')
            props_factor = {'type': 'factor', 'score': factor.get('gene_set_score', 1), 'name': label}
            id_factor = factor.get('factor')
            if log:
                logger.info("adding factor to graph with index: {}, label: {}".format(index, label))
            # will use string as key
            # graph.add_node("factor-{}".factor.get('label'), size=size, color=node_color, border_color=node_border_color, alpha=gene_node_opacity, label=node, gene=True)
            graph.add_node(id_factor, **props_factor)

        # get the extracted genes
        map_genes = dautils.extract_pigean_gene_factor_results_map(list_factor=list_factor, list_factor_genes=list_factor_genes, max_num_per_factor=max_num_genes)

        for gene, list_value in map_genes.items():
            id_node_gene = gene
            props_gene = {'type': 'gene', 'name': gene}

            # TODO - add gene score to gene node
            # add the gene node
            graph.add_node(id_node_gene, **props_gene)

            # add edge
            for factor_row in list_value:
                id_node_factor = factor_row.get('factor')
                props_gene_edge = {'score': factor_row.get('factor_score')}
                graph.add_edge(id_node_factor, id_node_gene, **props_gene_edge)

        # get the extracted gene sets
        map_gene_sets = dautils.extract_pigean_gene_set_factor_results_map(list_factor=list_factor, list_factor_gene_sets=list_factor_gene_sets, max_num_per_factor=max_num_gene_sets)

        for gene_set, list_value in map_gene_sets.items():
            id_node_gene_set = gene_set
            props_gene_set = {'type': 'gene set', 'name': gene_set}

            # TODO - add gene set score to gene set node
            # add the gene set node
            graph.add_node(id_node_gene_set, **props_gene_set)

            # add edge
            for factor_row in list_value:
                id_node_factor = factor_row.get('factor')
                props_gene_set_edge = {'score': factor_row.get('factor_score')}
                graph.add_edge(id_node_factor, id_node_gene, **props_gene_set_edge)

    # return
    return graph


def build_test_graph(log=False):
    '''
    builds a test graph to test the visual part
    '''

    # Create a graph
    graph = nx.karate_club_graph()    

    # log
    logger.info("return test karate graph: {}".format(graph))

    # return 
    return graph


def extract_nodes_edges_from_graph_old(graph, log=False):
    '''
    extracts the nodes and edges from the data
    '''
    # get the edges and nodes
    data = nx.node_link_data(graph)
    nodes = [{'id': node['id'], 'label': str(node['id'])} for node in data['nodes']]
    edges = [{'from': edge['source'], 'to': edge['target'], 'width': edge['weight']} for edge in data['links']]

    # return
    return nodes, edges


def extract_nodes_edges_from_graph(graph, log=False):
    '''
    extracts the nodes and edges from the data
    '''
    nodes = []
    edges = []

    for node, node_attrs in graph.nodes(data=True):
        node_data = {
            'id': node,
            'label': node_attrs.get('label', ''),
            'shape': node_attrs.get('shape', 'ellipse'),
            'color': node_attrs.get('color', 'gray'),
            'size': node_attrs.get('size', 10)
        }
        nodes.append(node_data)

    for source, target, edge_attrs in graph.edges(data=True):
        edges.append({
            'from': source, 
            'to': target,
            'color': edge_attrs.get('color', 'blue'),
            'size': edge_attrs.get('size', 5)
            })

    return nodes, edges


def extract_data_from_graph(graph, log=False):
    '''
    extracts the nodes and edges from the data
    '''
    # get the edges and nodes
    data = nx.node_link_data(graph)

    # return
    return data

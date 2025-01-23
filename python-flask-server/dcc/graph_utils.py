
# imports
import numpy as np
import networkx as nx
import json
from pyvis.network import Network
import dcc.dcc_utils as dutils 


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


def build_factor_graph(list_factor, test=True, log=False):
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
        # start a new graph
        GRAPH = nx.Graph()

        # get the colors needed for the number of factors
        num_colors = len(list_factor)
        if log:
            logger.info("getting color list for factors of size: {}".format(num_colors))
        list_colors = generate_distinct_colors(N=num_colors)

        # add the factors as squares


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


def extract_nodes_edges_from_graph(graph, log=False):
    '''
    extracts the nodes and edges from the data
    '''
    # get the edges and nodes
    data = nx.node_link_data(graph)
    nodes = [{'id': node['id'], 'label': str(node['id'])} for node in data['nodes']]
    edges = [{'from': edge['source'], 'to': edge['target'], 'width': edge['weight']} for edge in data['links']]

    # return
    return nodes, edges


def extract_data_from_graph(graph, log=False):
    '''
    extracts the nodes and edges from the data
    '''
    # get the edges and nodes
    data = nx.node_link_data(graph)

    # return
    return data

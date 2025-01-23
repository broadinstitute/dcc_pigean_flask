
# imports
import pytest 
import json 

import dcc.graph_utils as gutils
import dcc.dcc_utils as dutils

# constants

# logger
logger = dutils.get_logger(name=__name__)



# test methods
def test_generate_distinct_colors_orig():
    '''
    test generating new colors
    '''
    # initialize
    list_result = []

    # get the data
    list_result = gutils.generate_distinct_colors_orig(N=7)

    # log
    logger.info("got color list: {}".format(json.dumps(list_result)))

    # test
    assert list_result is not None
    assert len(list_result) == 7



def test_generate_distinct_colors():
    '''
    test generating new colors
    '''
    # initialize
    list_result = []

    # get the data
    list_result = gutils.generate_distinct_colors(N=7)

    # log
    logger.info("got color list: {}".format(json.dumps(list_result)))

    # test
    assert list_result is not None
    assert len(list_result) == 7


def test_blend_colors():
    '''
    test generating new colors
    '''
    # initialize
    list_result = []

    # get the data
    list_result = gutils.generate_distinct_colors(N=4)

    # log
    logger.info("got color list: {}".format(json.dumps(list_result)))

    # get blended colors
    list_blended = gutils.blend_colors(color_list=list_result, weights=[.25, .25, .2, .4])

    # log
    logger.info("got blended color list: {}".format(json.dumps(list_blended)))

    # test
    assert list_result is not None
    assert len(list_result) == 4
    assert list_blended is not None
    assert len(list_blended) == 4



# imports
import requests
import os 
import copy 

import dcc.dcc_utils as dutils 

# constants
logger = dutils.get_logger(__name__)
# ENV_LLM_KEY = os.environ.get('MARC_CHAT_KEY')
ENV_LLM_KEY = os.environ.get('CHAT_KEY')
LLM_KEY = None
if ENV_LLM_KEY:
    LLM_KEY = ENV_LLM_KEY


# methods
def get_list_factor_names_from_llm(list_labels, list_factor_gene_sets, log=True):
    '''
    will query the LLM to get factor names
    '''
    # initialize
    list_result = copy.deepcopy(list_labels)
    list_factor_prompts = []

    # log
    if log:
        logger.info("got list factor labels: {}".format(list_labels))
        logger.info("got list factor gene sets: {}".format(list_factor_gene_sets))

    # build the prompt
    for list_gene_sets in list_factor_gene_sets:
        list_temp = [item.get(dutils.KEY_APP_GENE_SET) for item in list_gene_sets]

        # log
        if log:
            logger.info("joining gene set list: {}".format(list_temp))

        # test to make sure each factor has at leat one gene set
        if any(item for item in list_temp):
            list_factor_prompts.append(",".join(list_temp[0:5]))
        else:
            list_factor_prompts.append(None)

    if LLM_KEY is not None and any(item for item in list_factor_prompts):
        # prompt = "Print a label to assign to each group: %s" % (" ".join(["%d. %s" % (j+1, ",".join(list_factor_gene_sets[j].get('gene_set'))) for j in range(len(list_factor_gene_sets))]))
        # prompt = "Print a label to assign to each group: %s" % (" ".join(["%d. %s" % (j+1, ",".join(list_factor_prompts))]))
        prompt = "Print a label to assign to each group: %s" % (" ".join(["%d. %s" % (j+1, list_factor_prompts[j]) for j in range(len(list_factor_prompts))]))

        # log
        if log:
            logger.info("LLM - Querying LMM with prompt: %s" % prompt)

        # query LLM
        response = query_lmm(prompt, LLM_KEY)
        if response is not None:
            try:
                responses = response.strip().split("\n")
                responses = [x for x in responses if len(x) > 0]
                if len(responses) == len(list_result):
                    for i in range(len(list_result)):
                        list_result[i][dutils.KEY_APP_GENE_SET] = responses[i]
                else:
                    raise Exception

            except Exception:
                logger.error("Couldn't decode LMM response %s; using simple label" % response)
                pass

        # log
        if log:
            logger.info("got list factor label original: {}".format("\n".join(list_factor_prompts)))
            logger.info("got list factor label results: {}".format(list_result))

    # return
    return list_result


##This function is for labelling clusters. Update it with your favorite LLM if desired
def query_lmm(query, auth_key=None):
    '''
    method to query LLM
    '''
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer %s' % auth_key,
    }

    json_data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {
                'role': 'user',
                'content': '%s' % query,
            },
        ],
    }
    try:
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data).json()
        if "choices" in response and len(response["choices"]) > 0 and "message" in response["choices"][0] and "content" in response["choices"][0]["message"]:
            return response["choices"][0]["message"]["content"]
        else:
            logger.info("LLM - LMM response did not match the expected format; returning none. Response: %s" % response); 
            return None
    except Exception:
        logger.info("LLM - LMM call failed; returning None"); 
        return None


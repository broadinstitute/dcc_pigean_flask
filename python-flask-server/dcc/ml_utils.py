

# imports
import requests
import os 
import copy 
from typing import Any, Dict, Optional, Tuple, List

import dcc.dcc_utils as dutils 

# constants
logger = dutils.get_logger(__name__)
# ENV_LLM_KEY = os.environ.get('MARC_CHAT_KEY')
ENV_LLM_KEY = os.environ.get('CHAT_KEY')
LLM_KEY = None
if ENV_LLM_KEY:
    LLM_KEY = ENV_LLM_KEY

# local ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")


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

    if True:
    # if LLM_KEY is not None and any(item for item in list_factor_prompts):
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


def get_list_factor_names_from_ollama_llm(
    list_labels: List[Dict[str, Any]],
    list_factor_gene_sets: List[List[Dict[str, Any]]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 24,
    timeout_s: Tuple[float, float] = (3.0, 60.0),
    log: bool = True
) -> List[Dict[str, Any]]:
    """
    Calls query_ollama_llm(...) once per factor, and writes the returned label into
    list_result[i][dutils.KEY_APP_GENE_SET].

    Expects query_ollama_llm(...) returns a dict like:
      {"summary": "..."}
    """

    list_result = copy.deepcopy(list_labels)

    if log:
        logger.info("got list factor labels: %s", list_labels)
        logger.info("got list factor gene sets: %s", list_factor_gene_sets)

    n = min(len(list_result), len(list_factor_gene_sets))

    for i in range(n):
        factor_gene_sets = list_factor_gene_sets[i] or []

        # pull gene set names/strings
        gene_sets = []
        for item in factor_gene_sets:
            gs = None
            try:
                gs = item.get(dutils.KEY_APP_GENE_SET)
            except Exception:
                gs = None
            if gs:
                gene_sets.append(str(gs))

        if log:
            logger.info("factor %d gene sets extracted: %s", i + 1, gene_sets)

        if not gene_sets:
            continue

        preview = ", ".join(gene_sets[:5])

        # log
        if log:
            logger.info("\n\nquerying ollama llm with gene sets: \n{}".format(preview))

        try:
            resp = query_ollama_llm(
                text_gene_sets=preview
            )
        except Exception as e:
            logger.exception("Ollama query failed for factor %d: %s", i + 1, e)
            continue

        if log:
            logger.info("got ollama LLM result: {}".format(resp))

        # ---- UPDATED PARSING: prefer resp['summary'] ----
        label_text = None
        if isinstance(resp, dict):
            # your wrapper
            if isinstance(resp.get("summary"), str):
                label_text = resp["summary"]
            # fallback shapes (optional)
            elif isinstance(resp.get("response"), str):
                label_text = resp["response"]
            elif isinstance(resp.get("message"), dict) and isinstance(resp["message"].get("content"), str):
                label_text = resp["message"]["content"]

        if not label_text or not label_text.strip():
            if log:
                logger.warning("No label returned for factor %d. Response: %r", i + 1, resp)
            continue

        # normalize common formatting: markdown bold, bullets, quotes, numbering
        label = label_text.strip()
        label = label.replace("**", "").strip()
        label = label.splitlines()[0].strip()
        label = label.lstrip("-•*").strip()
        label = label.strip("\"'` ").strip()

        # keep label short-ish if model gets chatty
        words = label.split()
        if len(words) > 8:
            label = " ".join(words[:8]).strip()

        if label:
            list_result[i][dutils.KEY_APP_GENE_SET] = label

        if log:
            logger.info("factor %d label result: %s", i + 1, label)

    if log:
        logger.info("final factor label results: %s", list_result)

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


def query_ollama_llm(
    text_gene_sets: str,
    model: str = DEFAULT_MODEL,
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout_s: Tuple[float, float] = (3.0, 60.0),  # (connect timeout, read timeout).
    log: bool = True
) -> Dict[str, Any]:
    """
    Generate a summary using Ollama's /api/generate.

    Returns a dict with fields:
      - ok: bool
      - summary: str (if ok)
      - error: str (if not ok)
      - details: any optional debug details
    """



    if not isinstance(text_gene_sets, str) or not text_gene_sets.strip():
        return {"ok": False, "error": "Input 'text' must be a non-empty string."}

    # Prompt for a small model: keep it short and explicit.
    # prompt = (
    #     "Summarize the following text in 3-6 bullet points. "
    #     "Be concise and preserve key facts.\n\n"
    #     f"TEXT:\n{text.strip()}\n"
    # )

    prompt = (
        "Create a concise biological label (2–6 words) for this gene-set group. Return ONLY the label."
        f"Gene sets: {text_gene_sets.strip()}\n"
    )

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }
    if max_tokens is not None:
        # Ollama options often accept num_predict (token prediction limit).
        payload["options"]["num_predict"] = int(max_tokens)

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": "Cannot connect to Ollama. Is it running? Try: `ollama serve`.",
            "details": {"url": url},
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error": "Ollama request timed out.",
            "details": {"timeout_s": timeout_s},
        }
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Request to Ollama failed: {e.__class__.__name__}"}

    # Non-200 responses: try to extract useful error details.
    if resp.status_code != 200:
        err_text = resp.text[:1000] if resp.text else ""
        # Ollama often returns JSON error bodies; parse if possible.
        try:
            err_json = resp.json()
        except ValueError:
            err_json = None

        return {
            "ok": False,
            "error": f"Ollama returned HTTP {resp.status_code}.",
            "details": {"body": err_json if err_json is not None else err_text},
        }

    # Parse JSON success payload
    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "error": "Ollama returned invalid JSON."}

    # Ollama /api/generate returns fields like: response, done, etc.
    summary = data.get("response")

    print("got response: {}\n\n".format(data))
    if not isinstance(summary, str) or not summary.strip():
        return {
            "ok": False,
            "error": "Ollama returned an empty response.",
            "details": {"data_keys": list(data.keys())},
        }

    return {"ok": True, "summary": summary.strip(), "details": {"model": model}}


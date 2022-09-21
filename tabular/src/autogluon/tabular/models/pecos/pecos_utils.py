import pandas as pd
import json
import re
import numpy as np


def clean_str(s):
    return re.sub('\W+', '_', str(s)) # Convert all non-alphanumeric (or _) chars to _

def read_pred_outfile(fn, k = 1):
    """
    Reads predictions output from PECOS
    :param fn --> 
    :param k --> beam size
    """
    df_pred = pd.DataFrame(
        [
            (
                tuple(t[0] for t in r['data'][:k]),
                tuple(t[1] for t in r['data'][:k])
            )
            for r in load_json_multi(fn)
        ],
        columns=['labels', 'scores']
    )
    return df_pred 

def format_predictions(df_pred):
    y_pred = []
    confidence = []
    for i,row in df_pred.iterrows():
        y_pred.append(row['labels'][0])
        confidence.append(row['scores'][0])
    return np.array(y_pred), np.array(confidence) 

def load_json_multi(fn):
    with open(fn) as f:
        for ln in f:
            yield json.loads(ln)


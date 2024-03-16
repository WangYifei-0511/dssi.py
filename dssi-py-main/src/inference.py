from joblib import dump, load
import pandas as pd
import numpy as np
from .data_processor import log_txf, remap_emp_length

def get_prediction(**kwargs):
    clf = load('/mount/src/dssi.py/dssi-py-main/src/mdl.joblib')
    features = load('/mount/src/dssi.py/dssi-py-main/src/raw_features.joblib')
    pred_df = pd.DataFrame(kwargs, index=[0])
    pred_df = log_txf(pred_df, ['annual_inc'])
    pred_df['emp_len'] = pred_df['emp_length'].map(remap_emp_length)
    pred = clf.predict(pred_df[features])
    return pred[0]

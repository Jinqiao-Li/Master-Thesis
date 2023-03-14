# import all the required packages
import sys
import os

import pandas as pd
import numpy as np
import json, csv

import torch
from datasets import load_dataset, load_metric
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score, hamming_loss
import warnings
warnings.filterwarnings("ignore")
from train import LitModel


gwa_labels=37
iwa_labels=332
dwa_labels=2085

MAX_LENGTH = 64
TEXT = 'Task_deepl_de'
LABEL = 'GWA_deepl_de'

# files
path = '../switchdrive/thesis/data/task_to_ALL_DE.csv'
train_file = '../switchdrive/thesis/data/task_train.csv'
val_file = '../switchdrive/thesis/data/task_val.csv'
test_file = '../switchdrive/thesis/data/task_test.csv'

# model
german_model = "deepset/gbert-base" 
job_model = "agne/jobGBERT"
multilingual_model = "bert-base-multilingual-cased" 
multi_job_model = '../switchdrive/thesis/model_ep_30'

root_dir = '../switchdrive/thesis/'
models_dir = root_dir+'trained_models/'

checkpoint_gbert = models_dir + 'deepl_de/german_model.ckpt'
checkpoint_job = models_dir + 'deepl_de/job_model.ckpt'
checkpoint_multilingual = models_dir + 'deepl_de/multilingual_model.ckpt'
checkpoint_multi_job = models_dir + 'deepl_de/multi_job.ckpt'

CHECKPOINT = checkpoint_job
MODEL = job_model

data_df = pd.read_csv(path, index_col=0)
train_df = pd.read_csv(train_file, index_col=0)
test_df = pd.read_csv(test_file, index_col=0)
y_encoded = LabelEncoder().fit(train_df[LABEL])


def get_prediction(model,tokenizer,text, top_n: int=5):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    # perform inference to our model
    logits = model(**inputs).logits
    # get output probabilities by doing softmax
    # sigmoid for multi-label
    probs = logits[0].softmax(0)
    
    #get the top_n candidates and corresponding prob as score
    value, indices = probs.topk(top_n, sorted=True)
    # results = [(id_.item(),round(val.item(),4)) for val,id_ in zip(value[0], indices[0])]
    results = [(id_.item(),round(val.item(),4)) for val,id_ in zip(value, indices)]
    results = [j for item in results for j in item ]
    return results

def apply_classify_on_df(model,tokenizer,df):
    """
    Apply a function and return multiple values so that you can create multiple columns, return a pd.Series with the values instead:
    Source: https://queirozf.com/entries/pandas-dataframes-apply-examples
    """
    df[['pred_la1','la1score','pred_la2','la2score','pred_la3','la3score','pred_la4','la4score','pred_la5','la5score']] = df.apply(lambda row: pd.Series(get_prediction(model,tokenizer,row.loc[TEXT])), axis=1)
    return df


# main
if __name__=="__main__":
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL) 
    new_model = LitModel.load_from_checkpoint(checkpoint_path=CHECKPOINT)   
    new_model.eval()
    print(MODEL, "MODEL reloaded from checkpoint: {0} !".format(CHECKPOINT))
    
    pred_df = apply_classify_on_df(new_model,tokenizer,test_df)
    
    pred_df['pred_la1'] = y_encoded.inverse_transform(pred_df['pred_la1'].astype(int)).tolist()
    pred_df['pred_la2'] = y_encoded.inverse_transform(pred_df['pred_la2'].astype(int)).tolist()
    pred_df['pred_la3'] = y_encoded.inverse_transform(pred_df['pred_la3'].astype(int)).tolist()
    pred_df['pred_la4'] = y_encoded.inverse_transform(pred_df['pred_la4'].astype(int)).tolist()
    pred_df['pred_la5'] = y_encoded.inverse_transform(pred_df['pred_la5'].astype(int)).tolist()
    
    # save m1 gbert model's predicted result
    outfile = root_dir+'pred_test/deepl_de/job.csv'
    pred_df.to_csv(outfile, header=True)
    
    
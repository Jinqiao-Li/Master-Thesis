# import all the required packages
import sys
import os

import pandas as pd
import numpy as np
import json, csv
import datasets

from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score,recall_score,precision_score
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import random
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset, load_metric

# Create a dataloading module as per the PyTorch Lightning Docs
class DataModule(pl.LightningDataModule):
  
      def __init__(self, tokenizer,train_file, val_file, test_file, batch_size,task,y_level):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.task = task
        self.y_level = y_level
        # self.num_examples = num_examples

      # Loads and splits the data into training, validation and test sets split with a ratio of 7/1/2 
      def prepare_data(self):
        # 16480, 2354, 4709
        self.train = pd.read_csv(self.train_file, index_col=0)
        self.val = pd.read_csv(self.val_file, index_col=0)
        self.test = pd.read_csv(self.test_file, index_col=0)
        # self.train, self.validate, self.test = np.split(self.data.sample(frac=1), [int(.7*len(self.data)), int(.8*len(self.data))])
        
      # encode the sentences using the tokenizer  
      def setup(self, stage):
        self.le = LabelEncoder().fit(self.train[self.y_level])
        self.train = encode_sentences(self.tokenizer, self.train[self.task], self.train[self.y_level], self.le)
        self.val = encode_sentences(self.tokenizer, self.val[self.task], self.val[self.y_level], self.le)
        self.test = encode_sentences(self.tokenizer, self.test[self.task], self.test[self.y_level], self.le)
            
      # Load the training, validation and test sets in Pytorch Dataset objects
      def train_dataloader(self):
        dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size )
        return train_data

      def val_dataloader(self):
        dataset = TensorDataset(self.val['input_ids'], self.val['attention_mask'], self.val['labels']) 
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data

      def test_dataloader(self):
        dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels']) 
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data


class LitModel(pl.LightningModule):
  # Instantiate the model
  def __init__(self,tokenizer, model, learning_rate=2e-5, ):
    super().__init__()
    self.tokenizer = tokenizer
    self.model = model
    self.learning_rate = learning_rate
    self.num_labels = model.num_labels
    

  # Do a forward pass through the model
  def forward(self, input_ids, **kwargs):
        
    return self.model(input_ids, **kwargs)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
    self.optim = optimizer
    return optimizer

  def training_step(self, batch, batch_idx):
    #print(batch, batch_idx)
    # batch
    input_ids, attention_mask = batch[0], batch[1]
    label = batch[2]
    
    # fwd
    outputs = self(input_ids, attention_mask=attention_mask,labels=label) # loss, logits
    logits = outputs[1]
    
    # loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    # Calculate the loss. The view(-1) operation flattens the tensor
    loss = loss_fct(logits.view(-1, self.model.num_labels), label.view(-1))
    self.log("train_loss", loss, prog_bar=True, logger=True)
    # return {"loss": loss, "predictions": outputs, "labels": labels}
    return {'loss': loss}

  def validation_step(self, batch, batch_idx):
    # batch
    input_ids, attention_mask = batch[0], batch[1]
    label = batch[2]
   
    outputs = self(input_ids, attention_mask=attention_mask,labels=label) # loss, logits
    logits = outputs[1]
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    val_loss = loss_fct(logits.view(-1, self.model.num_labels), label.view(-1))
    a, y_hat = torch.max(logits, dim=1)
    val_acc = accuracy_score(y_hat.cpu(), label.cpu())
    val_acc = torch.tensor(val_acc)

    self.log("val_loss", val_loss, prog_bar=True, logger=True)
    return {'val_loss': val_loss}

  def test_step(self, batch, batch_idx):
    input_ids, attention_mask = batch[0], batch[1]
    label = batch[2]
    
    outputs = self(input_ids, attention_mask=attention_mask,labels=label) # loss, logits
    logits = outputs[1]
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    test_loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))

    a, y_hat = torch.max(logits, dim=1)
    test_acc = accuracy_score(y_hat.cpu(), label.cpu())
    
    self.log("test_loss", test_loss, prog_bar=True, logger=True)
    return {'test_loss':test_loss}

  def test_epoch_end(self, outputs):
    avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

    tensorboard_logs = {'avg_test_loss': avg_loss, 'avg_test_acc': avg_test_acc}
    return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}



def encode_sentences(tokenizer, source_sentences, target, labelEncoder,
                     max_length=64, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence 
      Args: tokenizer - the specified tokenizer; source sentences are the tasks(de or en), targets are GWA/IWA/DWA types
      Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    tokenized_sentences = {}

    for sentence in source_sentences:
        encoded_dict = tokenizer.encode_plus(
              sentence,
              max_length=max_length,
              add_special_tokens = True, # Add '[CLS]' and '[SEP]'
              padding="max_length",
              truncation=True,
              return_tensors=return_tensors
          )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        #encoded_dicts.append(encoded_dict)

    # concatenate tensor with respect to dimension 0(rows)
    input_ids = torch.cat(input_ids, dim = 0)
    
    # padding
    attention_masks = torch.cat(attention_masks, dim = 0)
    
    y_encoded = labelEncoder.transform(target)
    y_labels = torch.tensor(y_encoded)
    
    
    batch = {
      "input_ids": input_ids,
      "attention_mask": attention_masks,
      "labels": y_labels,
    }

    return batch


# hyperparameters
gwa_labels=37
iwa_labels=332
dwa_labels=2085

hidden_dropout_prob = 0.3
learning_rate = 2e-5
weight_decay = 1e-2
epochs = 5
batch_size = 16

# models' path
german_model = "deepset/gbert-base" 
job_model = "agne/jobGBERT"
multilingual_model = "bert-base-multilingual-cased" 
multi_job = '/srv/scratch2/jinq/model_ep_30'


path = '/srv/scratch2/jinq/taskontology/task_to_ALL_DE.csv'
train_file = '/srv/scratch2/jinq/taskontology/task_train.csv'
val_file = '/srv/scratch2/jinq/taskontology/task_val.csv'
test_file = '/srv/scratch2/jinq/taskontology/task_test.csv'
root_dir = "/srv/scratch2/jinq/"
base_dir = root_dir + 'trained_models/'

# delect the model and classification level (the only chaging part)
model_path = job_model
num_label_level = gwa_labels


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path) 
model =  AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_label_level)


summary_data = DataModule(tokenizer, train_file=train_file, test_file=test_file, val_file=val_file,
                                 batch_size = batch_size, task='Task_de', y_level='GWA_de')
model = LitModel(learning_rate = learning_rate, tokenizer = tokenizer, model = model)

# main
if __name__=="__main__":
    
    checkpoint = ModelCheckpoint(
        save_last=True )
   
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=2, verbose=True, mode="min")

    trainer = pl.Trainer(
         gpus=1,
         max_epochs = 10,
         min_epochs = 1,
#         auto_lr_find = False,
         checkpoint_callback = checkpoint,
         progress_bar_refresh_rate = 500,
         callbacks=[early_stop_callback]
        )


    # Fit the instantiated model to the data
    trainer.fit(model, summary_data)
    print('path of best model:', trainer.checkpoint_callback.best_model_path)
    trainer.save_checkpoint('de_trained_models/0305/job_model.ckpt')
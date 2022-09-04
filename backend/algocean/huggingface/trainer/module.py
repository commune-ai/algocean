
import os, sys
sys.path.append(os.environ['PWD'])
import datasets 
import datetime
import transformers
from copy import deepcopy
from typing import Union, List
from copy import deepcopy
from algocean import BaseModule
import torch
import ray
from algocean.utils import dict_put
from datasets.utils.py_utils import asdict, unique_values
import datetime
import pyarrow
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ocean_lib.models.data_nft import DataNFT
import fsspec
import os
# from ipfsspec.asyn import AsyncIPFSFileSystem
from fsspec import register_implementation
import asyncio
import io
from algocean.ocean import OceanModule
from algocean.utils import Timer, round_sig


from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, load_dataset_builder, load_metric




class TrainerModule(BaseModule):
    default_cfg_path = 'huggingface.trainer.module'

    datanft = None
    default_token_name='token' 
    last_saved = None

    
    dataset = {}
    def __init__(self, config:dict=None, override:dict={}):
        BaseModule.__init__(self, config=config)
        self.algocean = OceanModule()
        self.web3 = self.algocean.web3
        self.ocean = self.algocean.ocean
        self.override_config(override)
        self.hub = self.get_object('huggingface.hub.module.HubModule')()
        self.load_state()

    def load_state(self):
        self.load_dataset()
        self.load_model()
        self.load_optimizer()
        self.load_metric()
        self.load_schedular()
        
    def load_dataset(self):
        dataset_class = self.get_object('huggingface.dataset.module.DatasetModule')
        override = dict(dataset=self.config.get('dataset'), load_dataset=True, client=self.client)
        self.dataset = dataset_class(override=override).dataset
        self.load_pipeline()
        self.dataloaders = self.get_dataloaders()
        # st.write(self.dataset.dataset)

    def split(self, split):
        return self.dataset[split]


    def load_pipeline(self):
        kwargs = self.config.get('pipeline')
        pipeline_class = self.import_object(path=kwargs.get('module'))
        pipeline = getattr(pipeline_class, kwargs['init']['fn'])(**kwargs['init'].get('kwargs', {}))
        self.pipeline = pipeline
        
        def pipeline_function(examples):
            return pipeline(examples["text"], padding="max_length", truncation=True)
        
        for split, dataset in self.dataset.items():
            dataset = dataset.map(pipeline_function, batched=True)
            dataset = dataset.remove_columns(["text"])
            dataset = dataset.rename_column("label", "labels")    
            dataset.set_format('torch')
            self.dataset[split] = dataset
    
    @property
    def device(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return device

    def load_model(self):
        kwargs = self.config.get('model')
        model_class = self.import_object(path=kwargs.get('module'))
        model = getattr(model_class, kwargs['init']['fn'])(**kwargs['init'].get('kwargs', {}))
        self.model = model
        self.model.to(self.device)


    def load_metric(self):
        self.metric = load_metric(self.config['metric'])
        
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)




    @property
    def num_epochs(self):
        for k in ['num_epochs', 'epochs']:
            v = self.config.get(k)
            if isinstance(v, int):
                return v
        
        raise Exception('Please specify the number of epochs in the config via num_epochs, or epochs')

    @property
    def num_training_steps(self):
        return self.num_epochs * len(self.dataloader['train'])
        

    def load_optimizer(self):
        kwargs = self.config.get('optimizer')
        optimizer_class = self.import_object(kwargs['module'])
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs['params'])

    @property
    def batch_size(self):
        return self.config.get('batch_size', 8) 

    @property
    def shuffle(self):
        return self.config.get('shuffle', True)
    

    def get_dataloaders(self):
        self.dataloader = {}
        for split in self.dataset.keys():
            self.dataloader[split] = DataLoader(self.dataset[split], shuffle=self.shuffle, batch_size=self.batch_size)
        return self.dataloader
    def load_schedular(self):
        kwargs = self.config['schedular']
        get_scheduler = self.import_object(kwargs['module'])
        self.scheduler = get_scheduler( optimizer=self.optimizer, num_training_steps=self.num_training_steps, **kwargs['params'])

    def train(self):
        progress_bar = tqdm(range(self.num_training_steps))
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in self.dataloader['train']:
                st.write({k: v.shape for k, v in batch.items()})
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                st.write(outputs)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                # progress_bar.update(1)



if __name__ == '__main__':
    import streamlit as st
    import numpy as np
    from algocean.utils import *
    module = TrainerModule()

    module.train()
    

import os, sys
sys.path.append(os.environ['PWD'])
import datasets 
import transformers
from copy import deepcopy
from typing import Union
from copy import deepcopy
from algocean import BaseModule
import torch
import ray
from algocean.utils import dict_put
from datasets.utils.py_utils import asdict, unique_values

import fsspec
import os
from ipfsspec.asyn import AsyncIPFSFileSystem
from fsspec import register_implementation
import asyncio
import io

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset

class DatasetModule(BaseModule, Dataset):
    default_cfg_path = 'huggingface.dataset.module'
    default_wallet_key = 'default'
    wallets = {}
    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
        self.load_dataset()

    @staticmethod
    def is_load_dataset_config(kwargs):
        '''
        check if dataset config is a valid kwargs for load_dataset
        '''
        key_type_tuples = [('path',str)]
        for k, k_type in key_type_tuples:
            if not isinstance(kwargs.get(k, k_type)):
                return False
        return True

        return  'path' in kwargs


    def load_dataset(self, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            config_kwargs = self.config.get('dataset') 
            if isinstance(config_kwargs, list):
                args = config_kwargs
            elif isinstance(config_kwargs, dict):
                kwargs = config_kwargs

        self.dataset = load_dataset(*args, **kwargs)
        return self.dataset
    
    


    def get_info(self, to_dict =True):

        def get_info_fn(ds, to_dict=to_dict):
            ds_info = deepcopy(ds.info)
            if to_dict:
                ds_info = asdict(ds_info)
            return ds_info
        
        if isinstance(self.dataset, list):
            ds_info = list(map(get_info_fn, self.dataset))
        elif isinstance(self.dataset, dict):
            ds_info = {k:get_info_fn(ds=v) for k,v in self.dataset.items()}
        elif isinstance(self.dataset, Dataset):
            ds_info  = get_info_fn(ds=self.dataset)

        return ds_info


    info = property(get_info)


    def save(self):
        return self.client.ipfs.save_dataset(path='/', dataset=self.dataset)

    def load_pipeline(self, *args, **kwargs):
        
        if len(args) + len(kwargs) == 0:
            kwargs = self.cfg.get('pipeline')
            assert type(kwargs) != None 
            if type(kwargs)  == str:
                transformer.AutoTokenizer.from_pretrained(kwargs) 
        else:
            raise NotImplementedError

if __name__ == '__main__':
    import streamlit as st
    
    st.write(Dataset.__init__)
    module = DatasetModule()
    # dataset, model, tokenizer = {}, {}, {}
    # module.load_dataset(path="glue", name="mrpc", split=['train', 'validation'])
    # st.write(module)
    # dataset2 = load_dataset("wikitext", "wikitext-103-v1", split='train')
    # model = AutoModel.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # st.write(asdict(module.dataset.info))
    # st.write(module)

    # st.write(module.dataset.save_to_disk('/tmp/bro'))

    st.write(module.client.ipfs.save_dataset(module.dataset.shard(num_shards=20, index=0), path='/'))
    # st.write(module.save())

    pass
    # with ray.init(address="auto",namespace="commune"):
    #     model = DatasetModule.deploy(actor={'refresh': False})
    #     sentences = ['ray.get(model.encode.remote(sentences))', 'ray.get(model.encoder.remote(sentences)) # whadup fam']
    #     print(ray.get(model.self_similarity.remote(sentences)))

        
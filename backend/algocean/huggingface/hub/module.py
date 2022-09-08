
import os, sys
sys.path.append(os.environ['PWD'])
import datasets 
import datetime
import pandas as pd
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
import plotly.express as px
from huggingface_hub import HfApi
from ocean_lib.models.data_nft import DataNFT
import fsspec
import os
from ipfsspec.asyn import AsyncIPFSFileSystem
from fsspec import register_implementation
import asyncio
import io
from algocean.ocean import OceanModule
from algocean.utils import Timer, round_sig, cache
from algocean.utils import *

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, load_dataset_builder

def get_module_function_schema(module, completed_only=False):
    cls = resolve_class(module)

    cls_function_names = get_functions(cls)
    schema_dict = {}
    for cls_function_name in cls_function_names:

        fn = getattr(cls, cls_function_name)
        if cls_function_name.startswith('__') and cls_function_name.endswith('__'):
            continue
        fn_schema = get_function_schema(fn)
        if not is_fn_schema_complete(fn_schema) and completed_only:
            continue
        schema_dict[cls_function_name] = get_function_schema(fn)

    return schema_dict


def cache(path='/tmp/cache.pkl', mode='memory'):

    def cache_fn(fn):
        def wrapped_fn(*args, **kwargs):
            cache_object = None
            self = args[0]

            
            if mode in ['local', 'local.json']:
                try:
                    cache_object = self.client.local.get_pickle(path, handle_error=False)
                except FileNotFoundError as e:
                    pass
            elif mode in ['memory', 'main.memory']:
                if not hasattr(self, '_cache'):
                    self._cache = {}
                else:
                    assert isinstance(self._cache, dict)
                cache_object = self._cache.get(path)
            force_update = kwargs.pop('force_update', False)
            if not isinstance(cache_object,type(None)) and not force_update:
                return cache_object
    
            cache_object = fn(*args, **kwargs)

            # write
            if mode in ['local']:

                self.client.local.put_pickle(data=cache_object,path= path)
            elif mode in ['memory', 'main.memory']:
                '''
                supports main memory caching within self._cache
                '''
                self._cache[path] = cache_object
            return cache_object
        return wrapped_fn
    return cache_fn
   



class HubModule(BaseModule, HfApi):
    default_cfg_path = 'huggingface.hub.module'
    datanft = None
    default_token_name='token' 
    last_saved = None



    
    dataset = {}
    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
        self.algocean = OceanModule()
        self.web3 = self.algocean.web3
        self.ocean = self.algocean.ocean
        self.hf_api = HfApi(self.config.get('hub'))


 
    @cache(path='/tmp/datasets.json', mode='local')
    def list_datasets(self,return_type = 'dict', filter_fn=None, *args, **kwargs):
        

        datasets = self.hf_api.list_datasets(*args,**kwargs)
        filter_fn = self.resolve_filter_fn(filter_fn=filter_fn)


        if return_type in 'dict':
            
            datasets = list(map(lambda x: x.__dict__, datasets))
            if filter_fn != None and callable(filter_fn):
                datasets = list(filter(filter_fn, datasets))

        elif return_type in ['pandas', 'pd']:
            datasets = list(map(lambda x: x.__dict__, datasets))
            df = pd.DataFrame(datasets)
            df['num_tags'] = df['tags'].apply(len)
            df['tags'] = df['tags'].apply(lambda tags: {tag.split(':')[0]:tag.split(':')[1] for tag in tags  }).tolist()
            for tag_field in ['task_categories']:
                df[tag_field] = df['tags'].apply(lambda tag:tag.get(tag_field) )
            df['size_categories'] = df['tags'].apply(lambda t: t.get('size_categories'))
            df = df.sort_values('downloads', ascending=False)
            if filter_fn != None and callable(filter_fn):
                df = self.filter_df(df=df, fn=filter_fn)
            return df
        else:
            raise NotImplementedError

    
        return datasets
    

    @staticmethod
    def resolve_filter_fn(filter_fn):
        if filter_fn != None:
            if callable(filter_fn):
                fn = filter_fn

            if isinstance(filter_fn, str):
                filter_fn = eval(f'lambda r : {filter_fn}')
        
            assert(callable(filter_fn))
        return filter_fn



    @cache(path='/tmp/models.json', mode='local')
    def list_models(self,return_type = 'pandas',filter_fn=None, *args, **kwargs):
        models = self.hf_api.list_models(*args,**kwargs)
       
        filter_fn = self.resolve_filter_fn(filter_fn=filter_fn)


        if return_type in 'dict':
            models = list(map(lambda x: x.__dict__, models))
            if filter_fn != None and callable(filter_fn):
                models = list(filter(filter_fn, models))

        elif return_type in ['pandas', 'pd']:

            models = list(map(lambda x: x.__dict__, models))
            models = pd.DataFrame(models)
            if filter_fn != None and callable(filter_fn):
                models = self.filter_df(df=models, fn=filter_fn)

        else:
            raise NotImplementedError

        return models



    @property
    def models(self):
        df = pd.DataFrame(self.list_models(return_type='dict'))
        return df
    @property
    def datasets(self):
        df = pd.DataFrame(self.list_datasets(return_type='dict'))
        return df

    @property
    def task_categories(self):
        return list(self.datasets['task_categories'].unique())
    @property
    def pipeline_tags(self): 
        df = self.list_models(return_type='pandas')
        return df['pipeline_tag'].unique()
    @property
    def pipeline_tags_count(self):
        count_dict = dict(self.models_df['pipeline_tag'].value_counts())
        return {k:int(v) for k,v in count_dict.items()}

    def streamlit_main(self):
        pipeline_tags_count = module.pipeline_tags_count
        fig = px.pie(names=list(pipeline_tags_count.keys()),  values=list(pipeline_tags_count.values()))
        fig.update_traces(textposition='inside')
        fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
        st.write(fig)
    def streamlit_sidebar(self):
        pass

    def streamlit(self):
        self.streamlit_main()
        self.streamlit_sidebar()


    def streamlit_datasets(self, limit=True):
        with st.expander('Datasets', True):
            st.write(self.datasets.iloc[0:limit])

    def streamlit_models(self, limit=100):
        with st.expander('Models', True):
            st.write(self.models.iloc[0:limit])

    def streamlit_dfs(self, limit=100):
        self.streamlit_datasets(limit=limit)
        self.streamlit_models(limit=limit)

    def dataset_tags(self, limit=10, **kwargs):
        df = self.list_datasets(limit=limit,return_type='pandas', **kwargs)
        tag_dict_list = df['tags'].apply(lambda tags: {tag.split(':')[0]:tag.split(':')[1] for tag in tags  }).tolist()
        tags_df =  pd.DataFrame(tag_dict_list)
        df = df.drop(columns=['tags'])
        return pd.concat([df, tags_df], axis=1)

    @staticmethod
    def filter_df(df, fn):
        indices =  df.apply(fn, axis=1)
        return df[indices]
if __name__ == '__main__':
    import streamlit as st
    import numpy as np


    module = HubModule()

    st.write(module.pipeline_tags)
    st.write(module.task_categories)




    st.write(px.histogram(x=module.datasets.query('num_tags>=0 & num_tags<10')['num_tags']))
    
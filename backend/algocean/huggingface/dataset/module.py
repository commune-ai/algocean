
import os, sys
sys.path.append(os.environ['PWD'])
import datasets 
import datetime
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

from algocean.ocean import OceanModule
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, load_dataset_builder

class DatasetModule(BaseModule, Dataset):
    default_cfg_path = 'huggingface.dataset.module'
    default_wallet_key = 'default'
    wallets = {}
    dataset = {}
    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
        self.load_dataset_factory()
        self.load_dataset_builder()
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



    def load_dataset_factory(self, path=None):
        if path == None:
            path = self.config['dataset']['path']
        self.dataset_factory = datasets.load.dataset_module_factory(path)

    def load_dataset_builder(self):
        self.dataset_builder = datasets.load.import_main_class(self.dataset_factory.module_path)



    def load_dataset(self, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            config_kwargs = self.config.get('dataset')
            split = config_kwargs.get('split', ['train'])
            if isinstance(split, str):
                split = [split]
            if isinstance(split, list):
                split = {s:s for s in split}
            
            config_kwargs['split'] = split
            if isinstance(config_kwargs, list):
                args = config_kwargs
            elif isinstance(config_kwargs, dict):
                kwargs = config_kwargs

        self.dataset = load_dataset(*args, **kwargs)


        return self.dataset
    
    def get_split(self, split='train'):
        return self.dataset[split]


    @staticmethod
    def list_datasets():
        return datasets.list_datasets()

    def get_info(self, to_dict =True):

        def get_info_fn(ds, to_dict=to_dict):
            ds_info = deepcopy(ds.info)
            if to_dict:
                ds_info = asdict(ds_info)
            return ds_info
        
        if isinstance(self.dataset, list):
            ds_info = list(map(get_info_fn, self.dataset))
        elif isinstance(self.dataset, dict):
            ds_info =  get_info_fn(list(self.dataset.values())[0])
        elif isinstance(self.dataset, Dataset):
            ds_info  = get_info_fn(ds=self.dataset)
        
        return ds_info


    info = property(get_info)

    def resolve_state_path(self, path:str=None):
        if path == None:
            path = self.config.get('state_path', None)
        
        return path
        
    @property
    def builder_configs(self):
        return {v.name:v for v in module.dataset_builder.BUILDER_CONFIGS}


    def save(self,mode:str='ipfs'):
        if mode == 'ipfs':
            path_split_map = {}
            for split, dataset in self.dataset.items():
                path_split_map[split] = self.client.ipfs.save_dataset(dataset)
            self.config['state_path'] = path_split_map
        else:
            raise NotImplementedError
    
        return self.config['state_path']




    def load(self, path:dict=None, mode:str='ipfs'):
        path = self.resolve_state_path(path=path)
        st.write(path)
        if mode == 'ipfs':
            dataset = {}
            for split, cid in path.items():
                dataset[split] = self.client.ipfs.load_dataset(cid)
            self.dataset = datasets.DatasetDict(dataset)
        else:
            raise NotImplementedError

        return self.dataset


    def load_pipeline(self, *args, **kwargs):
        
        if len(args) + len(kwargs) == 0:
            kwargs = self.cfg.get('pipeline')
            assert type(kwargs) != None 
            if type(kwargs)  == str:
                transformer.AutoTokenizer.from_pretrained(kwargs) 
        else:
            raise NotImplementedError

    @property
    def path(self):
        return self.config['dataset']['path']

    @property
    def builder_name(self):
        return self.path

    @property
    def dataset_name(self):
        return self.path

    @property
    def config_name(self):
        return self.config['dataset']['name']

    def create_asset(self, datatoken='token', services:list=[] ):
        self.algocean.create_asset(datanft=self.path, datatoken=datatoken,
                                 services=services )


    @property
    def url_files(self):
        split_path_map = self.save()
        
        url_files = self.algocean.create_files([{'hash': p, 'type': 'ipfs'} for p in split_path_map.values()])

        return url_files
    @property
    def additional_information(self):

        info_dict = {
            'organization': 'huggingface',
            'package': {
                        'name': 'datasets',
                        'version': datasets.__version__
                        },
            'info': self.info    
        }
        return info_dict

    def create_service(self,
                        name: str= None,
                        service_type: str= 'download',
                        files:list = None,
                        timeout = 180000,
                        **kwargs):

        if name == None:
            name = self.config_name
        if files == None:
            files = self.url_files

        name = '.'.join([name, service_type])
        return self.algocean.create_service(name=name,
                                            timeout=timeout,
                                            service_type=service_type, 
                                            description= self.info['description'],
                                            files=files,
                                             additional_information=self.additional_information) 



    @property
    def metadata(self):
        st.sidebar.write(self.info)
        metadata ={}
        metadata['name'] = self.path
        metadata['description'] = self.info['description']
        metadata['author'] = self.algocean.wallet.address
        metadata['license'] = self.info.get('license', "CC0: PublicDomain")
        metadata['categories'] = []
        metadata['tags'] = []
        metadata['additionalInformation'] = self.info
        metadata['type'] = 'dataset'

        current_datetime = datetime.datetime.now().isoformat()
        metadata["created"]=  current_datetime
        metadata["updated"] = current_datetime

        return metadata
    @property
    def split_info(self):
        return self.info['splits']


    def create_asset(self):
        self.algocean.create_datanft(name=self.dataset_name)
        metadata = self.metadata
        services = self.create_service()

        if not isinstance(services, list):
            services = [services]

        return self.algocean.create_asset(datanft=self.dataset_name, datatoken='token', metadata=metadata, services=services)

if __name__ == '__main__':
    import streamlit as st
    # st.write(Dataset.__init__)
    module = DatasetModule()
    module.algocean.load()
    # module.algocean.create_datanft(name=module.builder_name)
    # st.write(module.create_service().__dict__)
    st.write(module.create_asset())
    # # module.dataset.info.FSGSFS = 'bro'
    # load_dataset('glue')
    # st.write(module.info)
    # # st.write(module.save())
    # ds_builder = load_dataset_builder('ai2_arc')
    # with ray.init(address="auto",namespace="commune"):
    #     model = DatasetModule.deploy(actor={'refresh': False})
    #     sentences = ['ray.get(model.encode.remote(sentences))', 'ray.get(model.encoder.remote(sentences)) # whadup fam']
    #     print(ray.get(model.self_similarity.remote(sentences)))


# section 4 referencing section 4.18
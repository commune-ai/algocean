
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

    datanft = None
    default_token_name='token' 

    
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
                cid = self.client.ipfs.save_dataset(dataset)
                path_split_map[split] = self.client.ipfs.ls(cid)
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
        self.algocean.create_asset(datanft=self.dataset_name, datatoken=datatoken,
                                 services=services )

    @property
    def url_files_metadata(self):
        pass

    def get_url_data(self):
        split_path_map = self.save()

        file_index = 0
        url_files = []
        split_url_files_info = {}
        cid2index = {}
        for split, split_file_configs in split_path_map.items():
            split_url_files_info[split] = []
      
            for file_config in split_file_configs:
                
                cid = file_config['CID']
                if cid not in cid2index:
                    url_files.append(self.algocean.create_files({'hash': cid, 'type': 'ipfs'})[0])
                    cid2index[cid] = file_index
                    file_index += 1

                split_url_files_info[split].append(dict(
                    name=file_config['name'].split('/')[1],
                    type=file_config['type'],
                    size = file_config['size'],
                    file_index=cid2index[cid],
                    file_hash =self.algocean.web3.toHex((self.algocean.web3.keccak(text=cid)))
                ))

        st.write(split_url_files_info, url_files)

        self.url_files = url_files
        self.split_url_files_info = split_url_files_info
        # url_files = 

        return url_files, split_url_files_info
    @property
    def additional_information(self):

        info_dict = {
            'organization': 'huggingface',
            'package': {
                        'name': 'datasets',
                        'version': datasets.__version__
                        },
            'info': self.info, 
            'file_info': self.split_url_files_info
        }
        return info_dict

    def dispense_tokens(self,token=None):
        if token == None:
            token =self.default_token_name
        self.algocean.dispense_tokens(datatoken=token,
                                      datanft=self.datanft)

    def create_service(self,
                        name: str= None,
                        service_type: str= 'access',
                        files:list = None,
                        timeout = 180000,
                        price_mode = 'free',
                        **kwargs):

        self.datanft = self.algocean.get_datanft(self.dataset_name, create_if_null=True)
        datatoken = self.algocean.create_datatoken(name=kwargs.get('datatoken', self.default_token_name))
        if price_mode =='free':
            self.algocean.create_dispenser(datatoken=datatoken,datanft=self.datanft)
        else:
            raise NotImplementedError

        if name == None:
            name = self.config_name
        if files == None:
            files, split_files_metadata = self.get_url_data()

        name = '.'.join([name, service_type])
        return self.algocean.create_service(name=name,
                                            timeout=timeout,
                                            datatoken=datatoken,
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

    @property
    def asset(self):
        return self.algocean.get_asset(self.dataset_name)

    @property
    def services(self):
        return self.asset.services

    @staticmethod
    def get_cid_hash(file):
        cid =  os.path.basename(file).split('.')[0]
        cid_hash = module.algocean.web3.toHex((module.algocean.web3.keccak(text=cid)))
        return cid_hash

    def download(self, service=None, destination='bruh/'):
        if service == None:
            service = self.services[0]

        module.algocean.download_asset(asset=module.dataset_name, service=service,destination=destination )
        

        
        for split,split_files_info in service.additional_information['file_info'].items():
            for file_info in split_files_info:
                file_index = file_info['file_index']
                file_name = file_info['name']
                st.write(file_info)
                did = self.asset.did

        # /Users/salvatore/Documents/commune/algocean/backend/bruh/datafile.did:op:6871a1482db7ded64e4c91c8dba2e075384a455db169bf72f796f16dc9c2b780,0
        # st.write(destination)
        # og_path = self.client.local.ls(os.path.join(destination, 'datafile.'+did+f',{file_index}'))
        # new_path = os.path.join(destination, split, file_name )
        # self.client.local.makedirs(os.path.dirname(new_path), exist_ok=True)
        # self.client.local.cp(og_path, new_path)

        files = module.client.local.ls(os.path.join(destination, 'datafile.'+module.asset.did+ f',{0}'))
        st.write(dict(zip(list(map(self.get_cid_hash, files)), files)))
                
            
    def create_asset(self, price_mode='free', services=None):

        self.datanft = self.algocean.create_datanft(name=self.dataset_name)

        metadata = self.metadata
        if services==None:
            services = self.create_service(price_mode=price_mode)

        if not isinstance(services, list):
            services = [services]
            
        return self.algocean.create_asset(datanft=self.dataset_name, metadata=metadata, services=services)

if __name__ == '__main__':
    import streamlit as st
    # st.write(Dataset.__init__)
    module = DatasetModule()
    module.algocean.load()
    # module.algocean.create_datanft(name=module.builder_name)

    # st.write(module.create_asset())
    module.dispense_tokens()
    # st.write(module.algocean.get_balance(datatoken='token', datanft=module.dataset_name))
    # st.write(module.asset.services[0].additional_information['file_info'])
    # st.write(module.get_url_data())
    module.download( )
    module.algocean.save()
    # st.write(module.algocean.save())
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
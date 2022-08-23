


import fsspec
import os
from ipfsspec.asyn import AsyncIPFSFileSystem
from fsspec import register_implementation
import asyncio
import json
import pickle
import io
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
import os, sys
sys.path.append(os.getenv('PWD'))

import requests
from ipfspy.utils import parse_response

from algocean.client.local import LocalModule
from algocean import BaseModule


# register_implementation(IPFSFileSystem.protocol, IPFSFileSystem)
# register_implementation(AsyncIPFSFileSystem.protocol, AsyncIPFSFileSystem)

# with fsspec.open("ipfs://QmZ4tDuvesekSs4qM5ZBKpXiZGun7S2CYtEZRB3DYXkjGx", "r") as f:
#     print(f.read())

    
class EstuaryModule(BaseModule):
    default_cfg_path= 'client.estuary.module'
    
    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
        self.api_key = self.get_api_key(api_key = self.config.get('api_key'))
        self.local =  LocalModule()

    @staticmethod
    def get_api_key(api_key=None):
        if api_key == None:
            api_key = self.config.get('api_key')
        api_key = os.getenv(api_key)
        if api_key != None:
            # if the api_key is a env variable
            return api_key
        else:
            # if the api key is just a key itself (raw)
            assert isinstance(api_key, str)
            return env_api_key
   
   



    # %% ../nbs/02_estuaryapi.ipynb 4

    def est_get_viewer(
        api_key: str=None # Your Estuary API key
    ):

        api_key = self.resolve_api_key(api_key)
        "View your Estuary account details"
        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get('https://api.estuary.tech/viewer', headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 6
    # list pins
    def list_pins(
        api_key: str=None # Your Estuary API key
    ):
        "List all your pins"
    
        api_key = self.resolve_api_key(api_key)
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get('https://api.estuary.tech/pinning/pins', headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 7
    # add pin
    def add_pin(self,
        file_name: str, # File name to pin
        cid: str, # CID to attach
        api_key: str=None # Your Estuary API key

    ):
        "Add a new pin object for the current access token."
        api_key = self.resolve_api_key(api_key)
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        json_data = {
            'name': name,
            'cid': cid,
        }

        response = requests.post('https://api.estuary.tech/pinning/pins', headers=headers, json=json_data)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 8
    # get pin by ID
    def get_pin(self,
        pin_id: str, # Unique pin ID
        api_key: str=None # Your Estuary API key

    ):
        "Get a pinned object by ID"
        api_key = self.resolve_api_key(api_key)
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'https://api.estuary.tech/pinning/pins/{pin_id}', headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 9
    # replace pin by ID
    def replace_pin(self,
        pin_id: str, # Unique pin ID
        api_key: str=None # Your Estuary API key

    ):
        api_key = self.resolve_api_key(api_key)
        "Replace a pinned object by ID"
        
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.post(f'https://api.estuary.tech/pinning/pins/{pin_id}', headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 10
    # remove pin by ID
    def remove_pin(self,
        pin_id: str, # Unique pin ID
        api_key: str=None # Your Estuary API key

    ):
        "Remove a pinned object by ID"
        api_key = self.resolve_api_key(api_key)
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.delete(f'https://api.estuary.tech/pinning/pins/{pin_id}', headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 12
    # create new collection
    def create_collection(self,
        name: str, # Collection name
        description: str, # Collection description
        api_key: str= None # Your Estuary API key

    ):
        "Create new collection"
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        json_data = {
            'name': name,
            'description': description,
        }

        response = requests.post('https://api.estuary.tech/collections/create', headers=headers, json=json_data)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 13
    # add content
    def add_content(self,
        collection_id: str, # Collection ID
        data: list, # List of paths to data to be added
        cids: list, # List of respective CIDs
        api_key: str= None, # Your Estuary API key

    ):
        "Add data to Collection"
        
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        json_data = {
            'contents': data,
            'cids': cids,
            'collection': collection_id,
        }

        response = requests.post('https://api.estuary.tech/collections/add-content', headers=headers, json=json_data)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 14
    # list collections
    def list_collections(self,
        api_key: str=None # Your Estuary API key
    ):
        "List your collections"
        
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get('https://api.estuary.tech/collections/list', headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 15
    # list collection content
    def list_coll_content(self,
        collection_id: str, # Collection ID
        api_key: str=None # Your Estuary API key


    ):
        api_key = self.resolve_api_key(api_key)

        "List contents of a collection from ID"
        
        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        response = requests.get(f'https://api.estuary.tech/collections/content/{collection_id}', headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 16
    # FS list content of a path
    def list_content_path(self,
        collection_id: str, # Collection ID
        path: str, # Path in collection to list files from
        api_key: str=None # Your Estuary API key
    ):
        "List content of a path in collection"

        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        params = {
            'col': collection_id,
        }

        response = requests.get(f'https://api.estuary.tech/collections/fs/list?col=UUID&dir={path}', params=params, headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 17
    # FS add content to path
    def add_content_path(self,
        collection_id: str, # Collection ID
        path: str, # Path in collection to add files to
        api_key: str=None # Your Estuary API key

    ):
        "Add content to a specific file system path in an IPFS collection"
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        params = {
            'col': collection_id,
        }

        response = requests.post(f'https://api.estuary.tech/collections/fs/add?col=UUID&content=LOCAL_ID&path={path}', params=params, headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 19
    # add client safe upload key
    def add_key(self,
        api_key:str, # Your Estuary API key
        expiry:str='24h' # Expiry of upload key
    ):
        "Add client safe upload key"
        
        headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
        }

        params = {
            'perms': 'upload',
            'expiry': expiry,
        }

        response = requests.post('https://api.estuary.tech/user/api-keys', params=params, headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 20
    def add_data(self,
        path_to_file: str, # Path to file you want to upload
        api_key: str=None # Your Estuary API key
    ):
        "Upload file to Estuary"
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
        }

        files = {
            'data': open(path_to_file, 'rb'),
        }


        response = requests.post('https://api.estuary.tech/content/add', headers=headers, files=files)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 21
    # add CID
    def add_cid(self,
        file_name: str, # File name to add to CID
        cid: str, # CID for file
        api_key: str=None, # Your Estuary API key

    ):
        "Use an existing IPFS CID to make storage deals."
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        }

        json_data = {
            'name': file_name,
            'root': cid,
        }

        response = requests.post('https://api.estuary.tech/content/add-ipfs', headers=headers, json=json_data)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 22
    # add CAR
    def add_car(self,
        path_to_file: str, # Path to file to store
        api_key: str=None, # Your Estuary API key
    ):
        "Write a Content-Addressable Archive (CAR) file, and make storage deals for its contents."
        
        headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
        }
        api_key = self.resolve_api_key(api_key)


        with open(path_to_file, 'rb') as f:
            data = f.read()

        response = requests.post('https://api.estuary.tech/content/add-car', headers=headers, data=data)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 23
    # make deal with specific provider
    def make_deal(self,
        content_id: str, # Content ID on Estuary
        provider_id: str, # Provider ID
        api_key: str=None # Your Estuary API key

    ):
        api_key = self.resolve_api_key(api_key)

        "Make a deal with a storage provider and a file you have already uploaded to Estuary"
        
        headers = {
        'Authorization': f'Bearer {api_key}',
        }

        json_data = {
            'content': content_id,
        }

        response = requests.post(f'https://api.estuary.tech/deals/make/{provider_id}', headers=headers, json=json_data)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 24
    # data by CID
    def view_data_cid(self,
        cid: str, # CID
        api_key: str=None, # Your Estuary API key

    ):
        "View CID information"
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }
        
        response = requests.get(f'https://api.estuary.tech/content/by-cid/{cid}', headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 25
    # list data
    def list_data(self,
        api_key: str=None # Your Estuary API key
    ):
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }
        
        response = requests.get('https://api.estuary.tech/content/stats', headers=headers)
        return response, parse_response(response)

    # list deals
    def list_deals( self,
        api_key: str=None # Your Estuary API key
    ):
        # list deals
        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }
        
        response = requests.get('https://api.estuary.tech/content/deals', headers=headers)
        return response, parse_response(response)

    # get deal status by id

    def resolve_api_key(self, api_key):
        if api_key == None:
            api_key = self.api_key
        assert isinstance(api_key, str)
        return api_key

    
    def get_deal_status(self,
        deal_id: str, # Deal ID,
        api_key: str=None # Your Estuary API key
    ):
        "Get deal status by id"

        api_key = self.resolve_api_key(api_key)

        headers = {
        'Authorization': f'Bearer {api_key}',
        }
        
        response = requests.get(f'https://api.estuary.tech/content/status/{deal_id}', headers=headers)
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 28
    # get Estuary node stats
    @staticmethod
    def get_node_stats():
        "Get Estuary node stats"

        response = requests.get('https://api.estuary.tech/public/stats')
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 29
    # get on chain deal data
    @staticmethod
    def get_deal_data():
        "Get on-chain deal data"

        response = requests.get('https://api.estuary.tech/public/metrics/deals-on-chain')
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 30
    # get miner query ask
    @staticmethod
    def get_miner_ask(
        miner_id: str # Miner ID
    ):
        "Get the query ask and verified ask for any miner"
        
        response = requests.get(f'https://api.estuary.tech/public/miners/storage/query/{miner_id}')
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 31
    # get failure logs by provider
    @staticmethod
    def get_failure_logs(
        miner_id: str # Miner ID
    ):
        "Get all of the failure logs for a specific miner"
        
        response = requests.get(f'https://api.estuary.tech/public/miners/failures/{miner_id}')
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 32
    # get deal logs by provider
    @staticmethod
    def get_deal_logs(
        provider_id: str # Provider ID
    ):
        "Get deal logs by provider"
        
        response = requests.get(f'https://api.estuary.tech/public/miners/deals/{provider_id}')
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 33
    # get provider stats
    @staticmethod
    def get_provider_stats(
        provider_id: str # Provider ID
    ):
        "Get provider stats"
        
        response = requests.get(f'https://api.estuary.tech/public/miners/stats/{provider_id}')
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 34
    # list providers
    @staticmethod
    def list_providers():
        "List Estuary providers"
        
        response = requests.get('https://api.estuary.tech/public/miners')
        return response, parse_response(response)

    # %% ../nbs/02_estuaryapi.ipynb 36
    @staticmethod
    def get_data(
        cid: str, # Data CID
        path_name: str # Path and filename to store the file at
    ):
        "Download data from Estuary CID"
        
        url = f'https://dweb.link/ipfs/{cid}'
        response = requests.get(url, allow_redirects=True)  # to get content
        with open(path_name, 'wb') as f:
            f.write(response.content)
        return response, parse_response(response)


   
   
    @property
    def tmp_root_path(self):
        return f'/tmp/algocean/{self.id}'
    
    def get_temp_path(self, path):
        tmp_path = os.path.join(self.tmp_root_path, path)
        if not os.path.exists(self.tmp_root_path):
            self.local.makedirs(os.path.dirname(path), exist_ok=True)
        
        return tmp_path
    
    
    def save_model(self, model, path:str=None):

        
        # self.mkdir(path, create_parents=True)
        
        tmp_path = self.get_temp_path(path=path)
        model.save_pretrained(tmp_path)
        self.mkdirs(path)
        
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path,  recursive=True)
        
        return cid

    def save_tokenizer(self, tokenizer, path:str=None):

        
        # self.mkdir(path, create_parents=True)
        
        tmp_path = self.get_temp_path(path=path)
        tokenizer.save_pretrained(tmp_path)
        self.mkdirs(path)
        
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path,  recursive=True)
        
        return cid

    def load_tokenizer(self,  path:str):
        tmp_path = self.get_temp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        model = AutoTokenizer.from_pretrained(tmp_path)
        self.local.rm(tmp_path,  recursive=True)
        return model

    def load_model(self,  path:str):
        tmp_path = self.get_temp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        model = AutoModel.from_pretrained(tmp_path)
        # self.fs.local.rm(tmp_path,  recursive=True)
        return model

    def load_dataset(self, path):
        tmp_path = self.get_temp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        dataset = Dataset.load_from_disk(tmp_path)
        # self.fs.local.rm(tmp_path,  recursive=True)
        
        return dataset

    def save_dataset(self, dataset, path:str=None):
        tmp_path = self.get_temp_path(path=path)
        dataset = dataset.save_to_disk(tmp_path)
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        # self.fs.local.rm(tmp_path,  recursive=True)
        return cid
    
    def put_json(self, data, path='json_placeholder.pkl'):
        tmp_path = self.get_temp_path(path=path)
        self.local.put_json(path=tmp_path, data=data)
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path)
        return cid

    def put_pickle(self, data, path='/pickle_placeholder.pkl'):
        tmp_path = self.get_temp_path(path=path)
        self.local.put_pickle(path=tmp_path, data=data)
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path)
        return cid
    def get_pickle(self, path):
        return pickle.loads(self.cat(path))

    def get_json(self, path):
        return json.loads(self.cat(path))
    
    def force_put(self, lpath, rpath, max_trials=10):
        trial_count = 0
        cid = None
        while trial_count<max_trials:
            try:
                cid= self.put(lpath=lpath, rpath=rpath, recursive=True)
                break
            except fsspec.exceptions.FSTimeoutError:
                trial_count += 1
                print(f'Failed {trial_count}/{max_trials}')
                
        return cid

    @property
    def id(self):
        return type(self).__name__ +':'+ str(hash(self))

    @property
    def name(self):
        return self.id

if __name__ == '__main__':
    import ipfspy
    import streamlit as st

    
    module = EstuaryModule()
    st.write(module.name)



    # import torch


    dataset = load_dataset('glue', 'mnli', split='trainff')
    st.write(module.save_dataset(dataset=dataset))
    # # cid = module.put_pickle(path='/bro/test.json', data={'yo':'fam'})
    # # st.write(module.get_pickle(cid))

    # st.write(module.ls('/dog'))
    # st.write(module.ls('/'))
    # st.write(module..get_object('/tmp/test.jsonjw4ij6u'))


    # st.write(module.local.ls('/tmp/bro'))
    # # st.write(module.add_data('/tmp/bro/state.json'))
    # st.write(module.get_node_stats())
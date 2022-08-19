


import fsspec
import os
from ipfsspec.asyn import AsyncIPFSFileSystem
from fsspec import register_implementation
import asyncio
import io
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
import os, sys
sys.path.append(os.getenv('PWD'))

from algocean.client.local import LocalModule


# register_implementation(IPFSFileSystem.protocol, IPFSFileSystem)
# register_implementation(AsyncIPFSFileSystem.protocol, AsyncIPFSFileSystem)

# with fsspec.open("ipfs://QmZ4tDuvesekSs4qM5ZBKpXiZGun7S2CYtEZRB3DYXkjGx", "r") as f:
#     print(f.read())

class ClientModule:
    def __init__(self):
        self.local = fsspec.filesystem("file")
        self.fs = AsyncIPFSFileSystem()
    
register_implementation(AsyncIPFSFileSystem.protocol, AsyncIPFSFileSystem)

    
    
class IPFSModule(AsyncIPFSFileSystem):
    
    def __init__(self, config={}):
        AsyncIPFSFileSystem.__init__(self)
        self.local =  LocalModule()

    @property
    def tmp_root_path(self):
        return f'/tmp/algocean/{self.__name__}'
    
    def get_tmp_path(self, path):
        tmp_path = os.path.join(self.tmp_root_path, path)
        try:
            self.local.mkdir(tmp_path, create_parents=True)
        except FileExistsError:
            pass
        
        return tmp_path
    
    
    def save_model(self, model, path:str):

        
        # self.mkdir(path, create_parents=True)
        
        tmp_path = self.get_tmp_path(path=path)
        model.save_pretrained(tmp_path)
        self.mkdirs(path)
        
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path,  recursive=True)
        
        return cid

    def save_tokenizer(self, tokenizer, path:str):

        
        # self.mkdir(path, create_parents=True)
        
        tmp_path = self.get_tmp_path(path=path)
        tokenizer.save_pretrained(tmp_path)
        self.mkdirs(path)
        
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        self.local.rm(tmp_path,  recursive=True)
        
        return cid

    

    def load_tokenizer(self,  path:str):
        tmp_path = self.get_tmp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        model = AutoTokenizer.from_pretrained(tmp_path)
        self.local.rm(tmp_path,  recursive=True)
        return model


    
    def load_model(self,  path:str):
        tmp_path = self.get_tmp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        model = AutoModel.from_pretrained(tmp_path)
        # self.fs.local.rm(tmp_path,  recursive=True)
        return model


    def load_dataset(self, path):
        tmp_path = self.get_tmp_path(path=path)
        self.get(lpath=tmp_path, rpath=path )
        dataset = Dataset.load_from_disk(tmp_path)
        # self.fs.local.rm(tmp_path,  recursive=True)
        
        return dataset

    @staticmethod
    def save_dataset(dataset, path:str):
        tmp_path = self.get_tmp_path(path=path)
        dataset = dataset.save_to_disk(tmp_path)
        cid = self.force_put(lpath=tmp_path, rpath=path, max_trials=10)
        # self.fs.local.rm(tmp_path,  recursive=True)
        return cid



    
          
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



if __name__ == '__main__':
    import ipfspy
    import streamlit as st

    module = IPFSModule()
    st.write(module.local.put_object(path='/tmp/test.json', data={'yo':'fam'}))
    # st.write(module.ls('/'))
    st.write(module.local.get_object('/tmp/test.jsonjw4ij6u'))


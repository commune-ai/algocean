


import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))
import datasets

from algocean.client.ipfs import IPFSModule
from algocean.client.local import LocalModule
from algocean.client.s3 import S3Module
from algocean.client.estuary import EstuaryModule
from algocean.client.pinata import PinataModule
from algocean.client.rest import RestModule
from algocean.client.ray.module import RayModule
from copy import deepcopy
from algocean import BaseModule
class ClientModule(BaseModule):
    default_cfg_path = 'client'

    CLASS_CLIENT_DICT = dict(
        ipfs = IPFSModule,
        local = LocalModule,
        s3 = S3Module,
        estuary = EstuaryModule,
        pinata = PinataModule,
        rest = RestModule,
        ray=RayModule
        # ray = RayModule
    )
    client_names = list(CLASS_CLIENT_DICT.keys())
    registered_clients = []

    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)

        if config == None:
            config = {}
        self.config = config




        if isinstance(config, list):
            '''
            cleint: ['ray', 'rest']
            '''
            client_names = config
            for client_name in client_names:
                self.register_client(client=client_name)
            
        elif isinstance(config, dict):
            '''
            client: {ray: ray_kwargs, rest: rest_kwargs}
            '''
            if 'client' in config:
                client_configs =config.get('client')
                assert isinstance(client_configs, dict)
                
            if len(config) == 0:
                client_configs = {c:{} for c in self.client_names}
            for client_name, client_kwargs in config.items():
                self.register_client(client=client_name, **client_kwargs)

    def register_client(self, client, **kwargs):
        assert isinstance(client, str)
        assert client in self.client_names,f"{client} is not in {self.client_names}"
        setattr(self, client, self.CLASS_CLIENT_DICT[client](**kwargs))
        self.registered_clients.append(client)

    def get_registered_clients(self):
        return self.registered_clients
            

if __name__ == '__main__':
    import streamlit as st
    st.write(RayModule)
    module = ClientModule()
    st.write(module.__dict__)

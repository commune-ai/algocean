


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
    default_config_path = 'client'

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
    default_clients = list(CLASS_CLIENT_DICT.keys())
    registered_clients = {}

    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
        self.register_clients(clients=self.config.get('client', self.config.get('clients')))

    def register_clients(self, clients=None):
        if clients == None:
            # resort to list of default cleitns if none
            clients = self.default_clients
        elif type(clients) in [list, dict]:
            if len(clients) == 0:
                clients = self.default_clients
        

        

        if isinstance(clients, bool):
            if clients == True:
                clients = self.default_clients
            else:
                return

        if isinstance(clients, list):
            assert all([isinstance(c,str)for c in clients]), f'{clients} should be all strings'
            for client in clients:
                self.register_client(client=client)
        elif isinstance(clients, dict):
            for client, client_kwargs in clients.items():
                self.register_client(client=client, **client_kwargs)
        else:
            raise NotImplementedError(f'{clients} is not supported')

    def register_all_clients(self):
        self.register_clients()

            

    def register_client(self, client, **kwargs):
        if client in self.blocked_clients:
            return
        assert isinstance(client, str)
        assert client in self.default_clients,f"{client} is not in {self.default_clients}"
        client_module = self.CLASS_CLIENT_DICT[client](**kwargs)
        setattr(self, client,client_module )
        self.registered_clients[client] = client_module

    def remove_client(client):
        self.__dict__.pop(client)
        return client
    
    delete_client = rm_client= remove_client
    
    def remove_clients(clients):
        return [self.remove_client(client) for client in clients]
            
    delete_clients = rm_clients= remove_clients

    def get_registered_clients(self):
        return self.registered_clients

    @property
    def clients_config(self):
        return self.config.get('clients',self.config.get('client'), {})

    @property
    def blocked_clients(self):
        v = None
        for k in ['block', 'blocked', 'ignore']:
            v =  self.config.get('clients', {}).get('block')
            if v == None:
                continue
            elif isinstance(v, list):
                return v
            
        if v == None:
            v = []
        else:
            raise NotImplementedError(f"v: {v} should not have been here")

        return v

    ignored_clients = blocked_clients
if __name__ == '__main__':
    import streamlit as st
    st.write(RayModule)
    module = ClientModule()
    st.write(ClientModule._config())
    st.write(module.__dict__)

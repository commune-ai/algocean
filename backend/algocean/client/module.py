


import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))
import datasets

from algocean.client.ipfs import IPFSModule
from algocean.client.local import LocalModule
from algocean.client.s3 import S3Module
from algocean.client.estuary import EstuaryModule
from algocean.client.pinata import PinataModule

class ClientModule:
    def __init__(self, config={}):
        self.ipfs = IPFSModule()
        self.local = LocalModule()
        self.s3 = S3Module()
        self.estuary = EstuaryModule()
        self.pinata = PinataModule()

if __name__ == '__main__':
    module = ClientModule()
    print(module.__dict__)
    module.local.put_json(path='/tmp/algocean/bro.json', data={'bro': 1})
    module.local.put_json(path='/tmp/algocean/bro/bro.json', data={'bro': 1})

    st.write(module.local.stat('/tmp/algocean/bro.json', recursive=True))
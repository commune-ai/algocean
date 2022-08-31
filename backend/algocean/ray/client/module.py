
# Create Ocean instance
import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))
from algocean import BaseModule



class ClientModule(BaseModule):
    default_cfg_path = 'ray.client.module'
    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)


module = ClientModule.deploy(actor=True)
st.write(module.get_functions(module))


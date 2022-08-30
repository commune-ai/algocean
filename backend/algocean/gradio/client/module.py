


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from algocean import BaseModule
from algocean.utils import *



class GradioClientModule(BaseModule):
    default_cfg_path =  'gradio.client.module'






if __name__ == "__main__":
    import streamlit as st

    module = GradioClientModule()
    module.client.__dict__


    
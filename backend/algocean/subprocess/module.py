


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from algocean import BaseModule
from algocean.utils import *
import shlex
import subprocess

class SubprocessModule(BaseModule):
    default_config_path =  'subprocess.module'
    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)


    def submit(command):
        return self.run_command(command)
    
    @staticmethod
    def run_command(command:str):

        process = subprocess.Popen(shlex.split(command))
        st.write(process.__dict__)
        return process







if __name__ == "__main__":

    import streamlit as st
    module = SubprocessModule()
    st.write(module)
    st.write(module.run_command('python algocean/gradio/api/module.py --no-api --module="gradio.client.module.ClientModule"'))


    
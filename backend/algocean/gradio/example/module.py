


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from algocean import BaseModule
from inspect import getfile
import inspect
import socket
from algocean.utils import SimpleNamespace

class ExampleModule(BaseModule):
    default_config_path =  'gradio.example'
    def bro(self, input1=1, inpupt2=10, output_example={'bro': 1}):
        pass





import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from algocean import BaseModule
from algocean.utils import *
from multiprocessing import Process

class ProcessModule(BaseModule):
    def __init__(self, config=None):
        self.processes = []

    def submit(self, fn, *args, **kwargs):
        p = Process(target=fn, args=args, kwargs=kwargs)
        p.start()
        self.processes.append(p)

    def shutdown(self, p):
        pass

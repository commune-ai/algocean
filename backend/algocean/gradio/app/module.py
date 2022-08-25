


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from algocean import BaseModule
from inspect import getfile
import inspect
import socket
from algocean.utils import SimpleNamespace


class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    


class GradioModule(BaseModule):
    default_cfg_path =  'gradio.module'


    # without '__reduce__', the instance is unserializable.
    def __reduce__(self):
        deserializer = GradioModule
        serialized_data = (self.config,)
        return deserializer, serialized_data


    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)

        self.port2module = {} 
        self.module2port = {}
        self.host  = self.config.get('host', '0.0.0.0')
        self.port  = self.config.get('port', 8000)
        self.num_ports = self.config.get('num_ports', 10)
        self.port_range = self.config.get('port_range', [7860, 7865])
        

    @property
    def active_modules(self):
        return self._modules

    @property
    def gradio_modules(self):
        return self._modules

    def add_module(self, port, metadata:dict):
        self.port2module[port] = metadata
        # self.module2port[module]
        return True

    def rm_module(self, port:str=10, output_example=['bro']):
        visable.remove(current)
        return jsonify({"executed" : True,
                        "ports" : current['port']})


    @staticmethod
    def find_registered_functions(self):
        '''
        find the registered functions
        '''
        fn_keys = []
        for fn_key in GradioModule.get_funcs(self):
            try:
                if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioModule.register.__name__:
                    fn_keys.append(fn_key)
            except:
                continue
        return fn_keys


    @staticmethod
    def get_funcs(self):
        return [func for func in dir(self) if not func.startswith("__") and callable(getattr(self, func, None)) ]


    @staticmethod
    def has_registered_functions(self):
        '''
        find the registered functions
        '''
        for fn_key in GradioModule.get_funcs(self):
            if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioModule.register.__name__:
                return True


        return False



    def list_modules(self, mode='config', output_example=['bro']):

        assert mode in ['config', 'module']

        '''
        mode: options are active (running modules) all and inactive
        '''

        module_config_list = list(map( lambda x: x['config'], self.config_manager.module_tree(tree=False)))


        module_list = []
        for m_cfg_path in module_config_list:

            try:
                m_cfg = self.config_loader.load(m_cfg_path)

                object_module = self.get_object(m_cfg['module'])
                if self.has_gradio(object_module):
                    module_list.append(object_module)
            except:
                continue


        st.write(module_list)

        return module_list


    def active_port(self, port:int=1):
        is_active = port in self.port2module
        return is_active


    def portConnection(self ,port : int):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        result = s.connect_ex((self.host, port))
        if result == 0: return True
        return False


    @staticmethod
    def has_gradio(self):
        return GradioModule.has_registered_functions(self)



    def suggest_port(self, max_trial_count=10):
        trial_count = 0 
        for port in range(*self.port_range):
            print(port, 'port', not self.portConnection(port))
            if not self.portConnection(port):
                return port

        '''
        TODO: kill a port when they are all full
        '''
        raise Exception(f'There does not exist an open port between {self.port_range}')
        
    @staticmethod
    def compile(self, live=False, flagging='never', theme='default', **kwargs):
        print("Just putting on the finishing touches... 🔧🧰")
        for func in GradioModule.find_registered_functions(self):
            registered_fn = getattr(self, func)
            registered_fn()


        demos = []
        names = []
        for fn_key, fn_params in self.registered_gradio_functons.items():                
            names.append(func)
            demos.append(gradio.Interface(fn=getattr(self, fn_key),
                                        inputs=fn_params['inputs'],
                                        outputs=fn_params['outputs'],
                                        theme='default'))
            print(f"{fn_key}....{bcolor.BOLD}{bcolor.OKGREEN} done {bcolor.ENDC}")


        print("\nHappy Visualizing... 🚀")
        return gradio.TabbedInterface(demos, names)

    @staticmethod
    def register(inputs, outputs):
        def register_gradio(func):
               
            def wrap(self, *args, **kwargs):     
                if not hasattr(self, 'registered_gradio_functons'):
                    print("✨Initializing Class Functions...✨\n")
                    self.registered_gradio_functons = dict()

                fn_name = func.__name__ 
                if fn_name in self.registered_gradio_functons: 
                    result = func(self, *args, **kwargs)
                    return result
                else:
                    self.registered_gradio_functons[fn_name] = dict(inputs=inputs, outputs=outputs)
                    return None
            wrap.__decorator__ = GradioModule.register
            return wrap
        return register_gradio



import socket
import argparse
from fastapi import FastAPI
import uvicorn

app = FastAPI()


module = None


def get_module(config = {}):
    global module
    if module == None:
        module = GradioModule(config=config)
    
    return module



@app.get("/")
async def root():
    module = get_module()
    print(module)
    return {"message": "Hello World"}


@app.get("/append/port")
def append_port(port:int=10):
    module = get_module()
    current = request.json
    visable.append(port)
    return jsonify({"executed" : True})


@app.get("/append/fam")
def append_port(port:int=10):
    module = get_module()
    current = request.json
    visable.append(port)
    return jsonify({"executed" : True})


@app.put("/remove/port")
def remove_port(port:int=10, output_example:dict={'bro': {'bro':1}}):
    module = get_module()
    current = request.json
    print(current)
    visable.remove(current)
    return jsonify({"executed" : True,
                    "ports" : current['port']})

@app.get("/open/ports")



def open_ports():
    module = get_module()

    return jsonify(visable)



import inspect

def get_parents(cls):
    return list(cls.__mro__[1:-1])

def get_parent_functions(cls):
    parent_classes = get_parents(cls)
    function_list = []
    for parent in parent_classes:
        function_list += get_functions(parent)

    return list(set(function_list))




from algocean.utils import *




if __name__ == "__main__":
    import streamlit as st

    module = GradioModule()
    # st.write(type(GradioModule.active_port))
    import json


    st.write(get_function_schema(GradioModule.rm_module))



    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
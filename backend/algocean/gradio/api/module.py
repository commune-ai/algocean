


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from algocean import BaseModule
from inspect import getfile
import inspect
import socket
from algocean.utils import SimpleNamespace
from algocean.utils import *

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
    default_cfg_path =  'gradio.api.module'


    # without '__reduce__', the instance is unserializable.
    def __reduce__(self):
        deserializer = GradioModule
        serialized_data = (self.config,)
        return deserializer, serialized_data


    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
        print(self.client, 'CLIENTS')

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



    def get_modules(self, force_update=True):
        
        modules = []
        failed_modules = []
        print( self.client)
        for root, dirs, files in self.client.local.walk('/app/algocean'):
            if all([f in files for f in ['module.py', 'module.yaml']]):

                try:
                    
                    cfg = self.config_loader.load(root)   
                    if cfg == None:
                        cfg = {}           
                except TypeError as e:
                    cfg = {}



                module_path = cfg.get('module')
                if isinstance(module_path, str):
                    modules.append(module_path)
                elif module_path == None: 
                    failed_modules.append(root)

        return modules

    def get_gradio_modules(self):
        return list(self.get_module_schemas().keys())

    def get_module_schemas(self):
        module_schema_map = {}
        module_paths = self.get_modules()

        for module_path in module_paths:
            module = self.get_object(module_path)
            module_fn_schemas = get_full_functions(module)
            

            if len(module_fn_schemas)>0:
                module_schema_map[module_path] = module_fn_schemas
        

        return module_schema_map


    module = None
    @classmethod
    def get_module(cls, config = {}):
        if cls.module == None:
            cls.module = cls(config=config)
        return cls.module

import socket
import argparse
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    module = GradioModule.get_module()
    print(module)
    return {"message": "Hello World"}


@app.get("/modules")
async def modules():
    module = GradioModule.get_module()
    modules = module.get_modules()
    return modules

if __name__ == "__main__":
    # import streamlit as st

    # module = GradioModule()
    # # st.write(type(GradioModule.active_port))
    # import json
    # st.write(module.get_gradio_modules())
    # st.write(module.get_module_schemas())


                


    # st.write(module_list)
    # st.write(get_function_schema(GradioModule.rm_module))
    uvicorn.run("module:app", host="0.0.0.0", port=8000, reload=True, workers=2)
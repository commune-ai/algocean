import ray
from algocean.config import ConfigLoader
from algocean.ray.utils import create_actor, actor_exists, kill_actor, custom_getattr
from algocean.utils import dict_put, get_object, dict_get, get_module_file, get_function_defaults, get_function_schema, is_class, Timer, get_functions
import os
import numpy as np
import datetime
import inspect
from types import ModuleType
from importlib import import_module
class ActorModule: 
    config_loader = ConfigLoader(load_config=False)
    default_cfg_path = None
    def __init__(self, config=None):

        self.config = self.resolve_config(config=config)
        self.start_timestamp = datetime.datetime.utcnow().timestamp()

    def resolve_config(self, config, override={}, local_var_dict={}, recursive=True):
        if config == None:
            config = getattr(self,'config',  None)
        if config == None:
            assert isinstance(self.default_cfg_path, str)
            config = self.default_cfg_path

        config = self.load_config(config=config, 
                             override=override, 
                            local_var_dict=local_var_dict,
                            recursive=True)

        return config

    @staticmethod
    def load_config(config=None, override={}, local_var_dict={}, recursive=True):
        """
        config: 
            Option 1: dictionary config (passes dictionary) 
            Option 2: absolute string path pointing to config
        """
        return ActorModule.config_loader.load(path=config, 
                                    local_var_dict=local_var_dict, 
                                     override=override,
                                     recursive=True)


    @classmethod
    def default_cfg(cls, override={}, local_var_dict={}):

        return cls.config_loader.load(path=cls.default_cfg_path, 
                                    local_var_dict=local_var_dict, 
                                     override=override)

    @staticmethod
    def get_module(config, actor=False, override={}):
        """
        config: path to config or actual config
        client: client dictionary to avoid child processes from creating new clients
        """
        module_class = None
        # if this is a class return the class
        if is_class(config):
            module_class = config
            return module_class


        if isinstance(config, str):
            # check if object is a path to module, return None if it does not exist
            module_class = ActorModule.get_object(key=config, handle_failure=True)


        if isinstance(module_class, type):
            
            config = module_class.default_cfg()
       
        else:

            config = ActorModule.load_config(config)
            ActorModule.check_config(config)
            module_class = ActorModule.get_object(config['module'])

        return module_class.deploy(config=config, override=override, actor=actor)

    @staticmethod
    def check_config(config):
        assert isinstance(config, dict)
        assert 'module' in config



    @staticmethod
    def get_object(key, prefix = 'algocean', handle_failure= False):
        return get_object(path=key, prefix=prefix, handle_failure=handle_failure)

    @staticmethod
    def import_module(key):
        return import_module(key)

    @classmethod
    def deploy(cls, config=None, actor=False , override={}, local_var_dict={}):
        """
        deploys process as an actor or as a class given the config (config)
        """
        config = ActorModule.resolve_config(cls, config=config, local_var_dict=local_var_dict, override=override)

        if actor:
            config['actor'] = config.get('actor', {})
            if isinstance(actor, dict):
                config['actor'].update(actor)
            elif isinstance(actor, bool):
                pass
            else:
                raise Exception('Only pass in dict (actor args), or bool (uses config["actor"] as kwargs)')  
            
            return cls.deploy_actor(config=config, **config['actor'])
        else:
            return cls(config=config)

    @classmethod
    def deploy_actor(cls,
                        config,
                        name='actor',
                        detached=True,
                        resources={'num_cpus': 1, 'num_gpus': 0.1},
                        max_concurrency=1,
                        refresh=False,
                        verbose = True, 
                        redundant=False):
        return create_actor(cls=cls,
                        name=name,
                        cls_kwargs={'config': config},
                        detached=detached,
                        resources=resources,
                        max_concurrency=max_concurrency,
                        refresh=refresh,
                        return_actor_handle=True,
                        verbose=verbose,
                        redundant=redundant)

    def get(self, key):
        return self.getattr(key)

    def getattr(self, key):
        return custom_getattr(obj=self, key=key)

    def down(self):
        self.kill_actor(self.config['actor']['name'])

    @staticmethod
    def kill_actor(actor):
        kill_actor(actor)
        return f'{actor} killed'
    
    @staticmethod
    def actor_exists(actor):
        return actor_exists(actor)

    @staticmethod
    def get_actor(actor_name):
        return ray.get_actor(actor_name)

    @property
    def context(self):
        if self.actor_exists(self.actor_name):
            return ray.runtime_context.get_runtime_context()

    @property
    def actor_name(self):
        return self.config['actor']['name']
    

    @property
    def actor_handle(self):
        if not hasattr(self, '_actor_handle'):
            self._actor_handle = self.get_actor(self.actor_name)
        return self._actor_handle

    @property
    def module(self):
        return self.config['module']

    @property
    def name(self):
        return self.config.get('name', self.module)

    def mapattr(self, from_to_attr_dict={}):
        '''
        from_to_attr_dict: dict(from_key:str->to_key:str)

        '''
        for from_key, to_key in from_to_attr_dict.items():
            self.copyattr(from_key=from_key, to_key=to_key)

    def copyattr(self, from_key, to_key):
        '''
        copy from and to a desintatio
        '''
        attr_obj = getattr(self, from_key)  if hasattr(self, from_key) else None
        setattr(self, to, attr_obj)


    @staticmethod
    def is_hidden_function(fn):
        if isinstance(fn, str):
            return fn.startswith('__') and fn.endswith('__')
        else:
            raise NotImplemented(f'{fn}')


    @staticmethod
    def get_functions(object):
        obect = get_functions(object)

    @classmethod
    def functions(cls, return_type='str', **kwargs):
        functions =  get_functions(obj=cls, **kwargs)
        if return_type in ['str', 'string']:
            return functions
        
        elif return_type in ['func', 'fn','functions']:
            return [getattr(cls, f) for f in functions]
        else:
            raise NotImplementedError


    @classmethod
    def describe(cls, obj=None, streamlit=False, sidebar=True,**kwargs):
        if obj == None:
            obj = cls

        assert is_class(obj)

        fn_list = cls.functions(return_type='fn', **kwargs)
        
        fn_dict =  {f.__name__:f for f in fn_list}
        if streamlit:
            import streamlit as st
            for k,v in fn_dict.items():
                with (st.sidebar if sidebar else st).expander(k):
                    st.write(k,v)
        else:
            return fn_dict
        
        

    @classmethod
    def hasfunc(cls, key):
        fn_list = cls.functions()
        return bool(len(list(filter(lambda f: f==key, fn_list)))>0)

    @classmethod
    def filterfunc(cls, key):
        fn_list = cls.functions()
        ## TODO: regex
        return list(filter(lambda f: key in f, fn_list))


    @classmethod
    def get_module_filepath(cls):
        return inspect.getfile(cls)

    @classmethod
    def get_config_path(cls):
        path =  cls.get_module_filepath().replace('.py', '.yaml')
        assert os.path.exists(path), f'{path} does not exist'
        assert os.path.isfile(path), f'{path} is not a dictionary'
        return path

    @classmethod
    def parents(cls):
        return get_parents(cls)

    @staticmethod
    def timeit(fn, trials=1, time_type = 'seconds', timer_kwargs={} ,*args,**kwargs):
        
        elapsed_times = []
        results = []
        
        for i in range(trials):
            with Timer(**timer_kwargs) as t:
                result = fn(*args, **kwargs)
                results.append(result)
                elapsed_times.append(t.elapsed_time(return_type=time_type))
        return dict(mean=np.mean(elapsed_times), std=np.std(elapsed_times), trials=trials, results=[])


    

        
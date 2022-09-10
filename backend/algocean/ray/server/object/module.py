import ray
import os, sys
sys.path.append(os.getenv('PWD'))
from ray.util.queue import Queue
from algocean import BaseModule
"""

Background Actor for Message Brokers Between Quees

"""
from algocean.ray.utils import kill_actor, create_actor
from algocean.utils import *



class ObjectServerModule(BaseModule):
    default_cfg_path = 'ray.server.object'
    cache_dict= {}
    flat_cache_dict = {}

    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)

    def put(self,key,value):
        object_id = ray.put(value)
        self.cache_dict[key]= value
        return object_id
    def get(self, key, get=True):
        object_id= self.cache_dict.get(key)
        if isinstance(object_id, ray._raylet.ObjectRef):
            if get:
                return ray.get(object_id)
        return object_id

    def get_cache_state(self, key=''):
        object_id_list ={}
        key_path_list = {}
        return 

    @property
    def resolve_fn(fn):
        if isinstance(fn, str):
            fn =  eval(f'lambda x: {fn}')
        elif callable(fn):
            pass

        assert callable(fn)
        return fn

    def search(self, *args, **kwargs):
        return {k:self.cache_dict[k]for k in self.search_keys(*args, **kwargs)}
    
    def search_keys(self, key=None, filter_fn = None):
            
        if isinstance(key,str):
            filter_fn = lambda x: key in x
        elif callable(key):
            filter_fn = key
        elif key == None:
            filter_fn = lambda x: True


        return list(filter(filter_fn, list(self.cache_dict.keys())))


    def pop(self, key):
        object_id = dict_delete(input_dict=self.cache_dict, keys=key)

    def ls(self, key=''):
        return dict_get(input_dict=self.cache_dict, keys=key)
    def glob(self,key=''):
        return dict_get(input_dict=self.cache_dict, keys=key)


    def has(self, key):
        return dict_has(input_dict=self.cache_dict, keys=key)




if __name__ == "__main__":
    module = ObjectServerModule.deploy(actor={'refresh': False}, ray={'address': 'auto'})
    st.write(module)
    st.write(module.put.remote('hey fam', {'whadup'}))
    st.write(ray.get(module.search.remote()))
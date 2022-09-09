import ray
from ray.util.queue import Queue

from algocean import BaseModule
"""

Background Actor for Message Brokers Between Quees

"""
from algocean.ray.utils import kill_actor, create_actor

class ObjectServer(BaseModule):
    default_cfg_path = 'ray.server.object'
    cache= {}
    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
    def put(self,key,value):
        self.cache[key] = ray.put(value)
    def get(self, key):
        return self.cache.get(key)
    def pop(self, key):
        object_id = self.cache.pop(key)
        ray

    def ls(self):
        return list(self.cache.values())

    def has(self, key):
        return self.object.get()


    





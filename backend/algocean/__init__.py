
from algocean.utils import get_object
from algocean.config.loader import ConfigLoader
from algocean.ray.actor import ActorModule
class BaseModule(ActorModule):
    client = None
    default_cfg_path = None
    def __init__(self, config=None):
        ActorModule.__init__(self,config=config)
        self.config_loader = ConfigLoader()
        if config!=None:
            if len(config) == 0:
                config = None

        self.config = self.get_config(config=config)
        self.client = self.get_clients(self.config.get('client'))
        self.get_submodules()

    def get_clients(self, config=None):
        print(config, 'CONFIG')
        if config != None:
            return self.get_object('client.module.ClientModule')(config={})
            
    def get_config(self, config=None):
        if config == None:

            assert self.default_cfg_path != None
            print(self.default_cfg_path, 'BRO')
            config = self.config_loader.load(path=self.default_cfg_path)
        return config
    
    @staticmethod
    def get_object(key, prefix = 'algocean', handle_failure= False):
        return get_object(path=key, prefix=prefix, handle_failure=handle_failure)


    def get_submodules(self):
        submodule_configs = self.config.get('submodule',self.config.get('submodules'))
        
        if submodule_configs != None:
            assert isinstance(submodule_configs, dict)
            for submodule_name, submodule_config in submodule_configs.items():
                submodule_class = self.get_object(submodule_config['module'])
                submodule_instance = submodule_class(config=submodule_config)
                setattr(self, submodule_name, submodule_instance)
    
    


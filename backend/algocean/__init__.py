
from algocean.utils import get_object
from algocean.config.loader import ConfigLoader
from algocean.ray.actor import ActorModule
class BaseModule(ActorModule):
    client = None
    default_config_path = None
    def __init__(self, config=None, override={}):
        ActorModule.__init__(self,config=config, override=override)
        self.config_loader = ConfigLoader()
        if config!=None:
            if len(config) == 0:
                config = None

        self.config = self.get_config(config=config)
        self.client = self.get_clients(clients=self.config.get('client', self.config.get('clients')))
        self.get_submodules()

    def get_clients(self, clients={}):
        if self.config.get('name') == 'ClientModule':
            return
        client_module_class = self.get_object('client.module.ClientModule')
        # if isinstance(self, client_module_class):
        #     return
        
        config = client_module_class.default_config()
        config['clients'] = clients

        if isinstance(config, dict) :
            return client_module_class(config=config)
        elif isinstance(config, client_module_class):
            return config 
        else:
            raise NotImplementedError
            
    def get_config(self, config=None):
        if config == None:

            assert self.default_config_path != None
            config = self.config_loader.load(path=self.default_config_path)
        return config
    

    def get_submodules(self, submodule_configs=None):
        '''
        input: dictionary of modular configs
        '''
        if submodule_configs == None:
            submodule_configs = self.config.get('submodule',self.config.get('submodules',{}))
    
        assert isinstance(submodule_configs, dict)
        for submodule_name, submodule_config in submodule_configs.items():
            submodule_class = self.get_object(submodule_config['module'])
            submodule_instance = submodule_class(config=submodule_config)
            setattr(self, submodule_name, submodule_instance)




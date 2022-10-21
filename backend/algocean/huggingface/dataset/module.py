
import os, sys
sys.path.append(os.environ['PWD'])
import datasets 
import datetime
import transformers
from copy import deepcopy
from typing import Union, List
from copy import deepcopy
from algocean import BaseModule
import torch
import ray
from algocean.utils import dict_put
from datasets.utils.py_utils import asdict, unique_values
import datetime
import pyarrow

from ocean_lib.models.data_nft import DataNFT
import fsspec
import os
from ipfsspec.asyn import AsyncIPFSFileSystem
from fsspec import register_implementation
import asyncio
import io
from algocean.ocean import OceanModule
from algocean.utils import *

import ipfsspec
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, load_dataset_builder



TEST_DATASET_OPTIONS = ['glue','wikitext', 'blimp' ]
def check_kwargs(kwargs:dict, defaults:Union[list, dict], return_bool=False):
    '''
    params:
        kwargs: dictionary of key word arguments
        defaults: list or dictionary of keywords->types
    '''
    try:
        assert isinstance(kwargs, dict)
        if isinstance(defaults, list):
            for k in defaults:
                assert k in defaults
        elif isinstance(defaults, dict):
            for k,k_type in defaults.items():
                assert isinstance(kwargs[k], k_type)
    except Exception as e:
        if return_bool:
            return False
        
        else:
            raise e


class DatasetModule(BaseModule, Dataset):
    default_config_path = 'huggingface.dataset.module'

    datanft = None
    default_token_name='token' 
    last_saved = None

    
    dataset = {}
    def __init__(self, config:dict=None, override:dict={}):
        BaseModule.__init__(self, config=config, override=override)
        
        self.algocean = OceanModule(override={'network':self.config.get('network')})
        self.hub = self.get_object('huggingface.hub.module.HubModule')()
        self.override_config(override)
        self.load_state(**self.config.get('dataset'))

    @property
    def web3(self):
        return self.algocean.web3
    @property
    def ocean(self):
        return self.algocean.ocean
    @property
    def network(self):
        return self.algocean.network
    @property
    def set_network(self, *args, **kwargs):
        return self.algocean.set_network(*args, **kwargs)
        
    def set_default_wallet(self, key):
        return self.algocean.set_default_wallet(key)

    def load_builder(self, path):
        self.dataset_factory = self.load_dataset_factory(path=path)
        self.dataset_builder = self.load_dataset_builder(factory_module_path=self.dataset_factory.module_path) 

    def list_datasets(self, filter_fn=None, *args, **kwargs):
        kwargs['return_type'] = 'pandas'  
        df = self.hub.list_datasets(filter_fn=filter_fn, *args, **kwargs)
        return df    

    def load_state(self, path, name=None, split=['train'] ,**kwargs):
        
        self.load_builder(path=path)
        if name == None:
            name = self.list_configs(path = path, return_type='dict.keys')[0]

        self.config['dataset'] = dict(path=path, name=name, split=split)
        

        self.dataset = self.load_dataset(**self.config['dataset'])

    @staticmethod
    def is_load_dataset_config(kwargs):
        '''
        check if dataset config is a valid kwargs for load_dataset
        '''
        key_type_tuples = [('path',str)]
        for k, k_type in key_type_tuples:
            if not isinstance(kwargs.get(k, k_type)):
                return False
        return True

        return  'path' in kwargs

    @staticmethod
    def load_dataset_builder( path:str=None, factory_module_path:str=None):
        if factory_module_path == None:
            assert isinstance(path, str)
            factory_module = datasets.load.dataset_module_factory(path)
            factory_module_path = factory_module.module_path

        dataset_builder = datasets.load.import_main_class(factory_module_path)
        return dataset_builder

    @staticmethod
    def load_dataset_factory( path:str):
        return datasets.load.dataset_module_factory(path)


        
    def load_dataset(self, **kwargs):
        '''
        path: path to the model in the hug
        name: name of the config / flavor of the dataset
        split: the split of the model
        
        '''
        if kwargs.get('name') == None:
            assert kwargs.get('path') != None
            name = self.list_configs(path = kwargs['path'], return_type='dict.keys')[0]

        # ensure the checks pass
        check_kwargs(kwargs=kwargs, defaults=['split', 'name', 'path' ])


        if len(kwargs) == 0:
            kwargs = self.config.get('dataset')

        split = kwargs.get('split', ['train'])
        
        if isinstance(split, str):
            split = [split]
        if isinstance(split, list):
            kwargs['split'] = {s:s for s in split}



        return  load_dataset(**kwargs )
    
    def get_split(self, split='train'):
        return self.dataset[split]





    def get_info(self, to_dict =True):

        def get_info_fn(ds, to_dict=to_dict):
            ds_info = deepcopy(ds.info)
            if to_dict:
                ds_info = asdict(ds_info)
            return ds_info
        
        if isinstance(self.dataset, list):
            ds_info = list(map(get_info_fn, self.dataset))
        elif isinstance(self.dataset, dict):
            ds_info =  get_info_fn(list(self.dataset.values())[0])
        elif isinstance(self.dataset, Dataset):
            ds_info  = get_info_fn(ds=self.dataset)
        
        return ds_info


    info = property(get_info)

    def resolve_state_path(self, path:str=None):
        if path == None:
            path = self.config.get('state_path', None)
        
        return path
        
    # @property
    # def builder_configs(self):
    #     return {v.name:v for v in module.dataset_builder.BUILDER_CONFIGS}

    @property
    def state_path_map(self):
        state_path_map = self.config.get('state_path_map')
        if state_path_map == None:
            state_path_map = self.save()
            self.config['state_path_map'] = state_path_map

        return state_path_map

    __file__ = __file__
    @property
    def local_tmp_dir(self):
        return f'/tmp/{self.module_path()}/{self.path}'



    def save(self,mode:str='estuary', chunks=20, chunk_index = 1 ):

        if mode == 'estuary':
            state_path_map = {}
            for split, dataset in self.dataset.items():
                split_path = os.path.join(self.local_tmp_dir, split)
                dataset.shard(chunks,chunk_index).save_to_disk(split_path)
                split_state = self.client.estuary.add(split_path)
                state_path_map[split] = {f.replace(split_path+'/',''): cid for f,cid in split_state.items()}

        elif mode == 'ipfs':
            state_path_map = {}
            for split, dataset in self.dataset.items():
                split_path = os.path.join(self.local_tmp_dir, split)
                dataset.shard(chunks,chunk_index).save_to_disk(split_path)
                split_state = self.client.ipfs.add(split_path)
                state_path_map[split] = {f.replace(split_path+'/',''): cid for f,cid in split_state.items()}

        elif mode == 'pinata':
            raise NotImplementedError

        else:
            raise NotImplementedError

        self.config['state_path_map'] = state_path_map

        self.last_saved = datetime.datetime.utcnow().timestamp()
    
        return state_path_map



    def load(self, path:dict=None, mode:str='ipfs'):
        path = self.resolve_state_path(path=path)
        if mode == 'ipfs':
            dataset = {}
            for split, cid in path.items():
                dataset[split] = self.client.ipfs.load_dataset(cid)
            self.dataset = datasets.DatasetDict(dataset)
        else:
            raise NotImplementedError

        return self.dataset


    def load_pipeline(self, *args, **kwargs):
        
        if len(args) + len(kwargs) == 0:
            kwargs = self.cfg.get('pipeline')
            assert type(kwargs) != None 
            if type(kwargs)  == str:
                transformer.AutoTokenizer.from_pretrained(kwargs) 
        else:
            raise NotImplementedError

    @property
    def path(self):
        return self.config['dataset']['path'].replace('/', '_')

    builder_name = dataset_name = path


    @property
    def url_files_metadata(self):
        pass

    @property
    def url_files(self):
    
        url_files = []
        file_index = 0
        cid2index = {}
        for split, split_file2cid in self.state_path_map.items():
            

            for filename, cid in split_file2cid.items():


                if cid not in cid2index:
                    url_files.append(self.algocean.create_files({'hash': cid, 'type': 'ipfs'})[0])
                    cid2index[cid] = file_index
                    file_index += 1
            

        return url_files


    def hash(self, data:str, algo='keccak'):
        if algo == 'keccak':
            return self.algocean.web3.toHex((self.algocean.web3.keccak(text=data)))
        else:
            raise NotImplementedError


    def get_split_file_info(self):

        file_index = 0
        url_files = []
        split_url_files_info = {}
        cid2index = {}
        url_files = deepcopy(self.url_files)
        for split, split_file2cid in self.state_path_map.items():
            split_url_files_info[split] = []
      
            for filename, cid in split_file2cid.items():
                
                file_index = None
                for i in range(len(url_files)):
                    if url_files[i].hash == cid:
                        file_index = i
                        break  
                    
                assert file_index != None

                split_url_files_info[split].append(dict(
                    name=filename ,
                    file_index=file_index,
                    cid=cid,
                    file_hash = self.hash(cid))
                )



        # url_files = 

        return split_url_files_info
    

    

    def additional_information(self, mode='service'):



        info_dict = {
            'organization': 'huggingface',
            'package': {
                        'name': 'datasets',
                        'version': str(datasets.__version__)
                        },
            # 'info': {'configs': self.configs}, 
        }

        if mode == 'service':
            info_dict['info'] = self.info
        elif mode == 'asset':
            info_dict['info'] = self.info
        else:
            raise NotImplementedError


        return info_dict

    def dispense_tokens(self,token=None, wallet=None):
        wallet = self.algocean.get_wallet(wallet)
        if token == None:
            token =self.default_token_name
        self.algocean.get_datatoken(datanft=self.datanft, datatoken=self.default_token_name)
        self.algocean.dispense_tokens(datatoken=token,
                                      datanft=self.datanft)


    def assets(self):
        return self.algocean.get_assets()





    def create_service(self,
                        name: str= None,
                        service_type: str= 'access',
                        files:list = None,
                        timeout = 180000,
                        price_mode = 'free',
                        **kwargs):

        datanft = self.datanft
        if datanft == None:
            datanft = self.algocean.create_datanft(name=self.dataset_name)
        datatoken = self.algocean.create_datatoken(datanft = self.datanft , name=kwargs.get('datatoken', self.default_token_name))
        if price_mode =='free':
            self.algocean.create_dispenser(datatoken=datatoken,datanft=self.datanft)
        else:
            raise NotImplementedError

        if name == None:
            name = self.config_name
        if files == None:
            files = self.url_files
        name = '.'.join([name, service_type])

        service = self.get_service(name)
        if service != None:
            return service
        else:
            return self.algocean.create_service(name=name,
                                                timeout=timeout,
                                                datatoken=datatoken,
                                                datanft = datanft,
                                                service_type=service_type, 
                                                description= self.info['description'],
                                                files=files,
                                                additional_information=self.additional_information('service')) 

    def get_service(self, name):
        for service in self.services:
            if service.name == name:
                return service
        return None


    @property
    def metadata(self):
        metadata ={}
        metadata['name'] = self.path
        metadata['description'] = self.info['description']
        metadata['author'] = self.algocean.wallet.address
        metadata['license'] = self.info.get('license', "CC0: PublicDomain")
        metadata['categories'] = []
        metadata['tags'] = [f'{k}:{v}' for k,v in self.tags.items()]
        metadata['additionalInformation'] = self.additional_information('asset')
        metadata['type'] = 'dataset'

        current_datetime = datetime.datetime.now().isoformat()
        metadata["created"]=  current_datetime
        metadata["updated"] = current_datetime

        return metadata
    
    
    @property
    def split_info(self):
        return self.info['splits']

    @property
    def features(self):
        return self.info['features']

    @property
    def builder_configs(self):
        return [c.__dict__ for c in self.dataset_builder.BUILDER_CONFIGS]

    @property
    def all_configs(self):
        return self.builder_configs

    @property
    def datatokens(self):
        return self.get_datatokens(asset=self.asset)


    @property
    def wallet(self):
        return self.algocean.wallet

    @property
    def wallets(self):
        return self.algocean.wallets

    @property
    def services(self):
        if self.asset == None:
            return []
        return self.asset.services


    @property
    def all_assets(self):
        assets =  self.algocean.search(text=f'')
        if len(assets) == 0:
            return self.create_asset()
        
        return assets[0]

    def search_assets(self, search='metadata'):
        return self.algocean.search(text=search)


    @property
    def my_assets(self):
        return self.algocean.search(text=f'metadata.author:{self.wallet.address}')

    @property
    def my_assets_info(self):
        return self.asset2info(self.my_assets)

    @staticmethod
    def asset2info(assets:list): 
        asset_info_list = []
        for a in assets:
            get_map = {
                'name': 'metadata.name',
                'organization': 'metadata.additionalInformation.organization',
                'did': 'did',
                'chain_id': 'chain_id'
            }
            'https://metahub.algovera.ai/asset/'
            asset_dict = a.__dict__
            asset_info_dict = {k: dict_get(asset_dict, v) for k,v in get_map.items()}
            asset_info_dict['url'] = f'https://metahub.algovera.ai/asset/{asset_info_dict["did"]}'
            asset_info_list.append(asset_info_dict)
        return asset_info_list


    @property
    def asset(self):
        assets =  self.algocean.search(text=f'metadata.name:{self.dataset_name} AND metadata.author:{self.wallet.address}')
        
        if len(assets) == 0:
            return None
        indices = list(map(int, list(np.argsort([a.metadata['created'] for a in assets]))))
        # get the most recent created asset
        len(assets)
        return assets[indices[-1]]

    @property
    def datanft(self):
        if not hasattr(self,'_datanft'):
            datanft =  self.algocean.get_datanft(self.asset.nft['address'])
            if datanft == None:
                datanft = self.algocean.create_datanft(name=self.dataset_name)
            
            assert isinstance(datanft, DataNFT)
            self._datanft = datanft


        return self._datanft 

    @datanft.setter
    def datanft(self, value):
        self._datanft = value
        



    @staticmethod
    def get_cid_hash(file):
        cid =  os.path.basename(file).split('.')[0]
        cid_hash = module.algocean.web3.toHex((module.algocean.web3.keccak(text=cid)))
        return cid_hash

    def get_service(self, service):
        services = self.services
        if isinstance(service, int):
            service = services[service]
        
        elif isinstance(service, str):
            services = [s for s in services if s.name == service]
            # assert len(services)==1, f'{service} is not in {services}'
            if len(services)>0:
                service = services[0]
            else:
                service=None
        else:
            return services[0]
            raise NotImplementedError
        
        return service

        
    def download(self, asset=None, service=None, destination='fam'):
        if destination[-1] != '/':
            destination += '/'
        
        if service == None:
            service = self.get_service(service)
        datatoken = self.algocean.get_datatoken(service.datatoken)
        
        if datatoken.balanceOf(self.wallet.address)< self.ocean.to_wei(1):
            self.dispense_tokens()
        self.algocean.download_asset(asset=self.asset, service=service,destination=destination )
        
        did_folder = f'datafile.{self.asset.did},0'
        download_dir = f'{destination}/{did_folder}'
        download_files = self.client.local.ls(download_dir)
        splits_info = dict_get(service.__dict__, 'additional_information.info.splits')
        path2hash = {f:self.hash(os.path.basename(f).split('.')[0]) for f in download_files}
        path2path = {}
        for split, split_info in splits_info.items():
            file_info_list = split_info['file_info']
            for file_info in file_info_list:
                for f, h in path2hash.items():
                    if h == file_info['file_hash']:
                        new_f = deepcopy(f)
                        new_f = f.replace(f'/{did_folder}/', '/').replace(os.path.basename(f), file_info['name'] )
                        path2path[f] = new_f
        for p1, p2 in path2path.items():
            self.client.local.cp(p1, p2)
                 
        self.client.local.rm(download_dir, recursive=True)
        return self.load_from_disk(path=destination)

         

    def balanceOf(self, datatoken):
        self.aglocean.balanceOf
  
    def create_asset(self, price_mode='free', services=None, force_create = False):

        asset = self.asset
        if asset != None and force_create==False:
            return asset
        self.datanft = self.algocean.create_datanft(name=self.dataset_name)

        metadata = self.metadata
        if services==None:
            services = self.create_service(price_mode=price_mode)



        if price_mode == 'free':
            self.dispense_tokens()

        if not isinstance(services, list):
            services = [services]

        
            

        return self.algocean.create_asset(datanft=self.datanft, metadata=metadata, services=services)




    def load(self, path:dict=None, mode:str='ipfs'):
        path = self.resolve_state_path(path=path)
        if mode == 'ipfs':
            dataset = {}
            for split, cid in path.items():
                dataset[split] = self.client.ipfs.load_dataset(cid)
            self.dataset = datasets.DatasetDict(dataset)
        else:
            raise NotImplementedError

        return self.dataset




    def list_configs(self, path:str=None, return_type='dict'):
        if isinstance(path, str):
            if path != self.path:
                dataset_builder = self.load_dataset_builder(path)
            else:
                if hasattr(self, 'dataset_builder'):
                    dataset_builder = self.dataset_builder
                else:
                    dataset_builder = self.load_dataset_builder(path)
        else:
            dataset_builder = self.dataset_builder


        configs = [config.__dict__ for config in dataset_builder.BUILDER_CONFIGS]

        if len(configs) == 0:
            configs =  [dataset_builder('default').info.__dict__]
            configs[0]['name'] = 'default'

        if return_type.startswith('dict'):
            configs = {config['name']: config for config in configs}
            if return_type in ['dict.keys']:
                return list(configs.keys())
            elif return_type in ['dict.values']:
                return list(configs.values())
            elif return_type == 'dict':
                return configs
            else:
                raise NotImplementedError

        
        elif return_type in ['list']:
            return configs
        else:
            raise NotImplementedError
        return configs
    

    @staticmethod
    def timeit(fn, trials=1, time_type = 'seconds', return_results = False, timer_kwargs={} ,*args,**kwargs):
        
        elapsed_times = []
        results = []
        
        for i in range(trials):
            with Timer(**timer_kwargs) as t:
                result = fn(*args, **kwargs)
                results.append(result)
                elapsed_times.append(t.elapsed_time(return_type=time_type))
        
        stat_dict = {
            'mean':round_sig(np.mean(elapsed_times),3), 
                'std':round_sig(np.std(elapsed_times),3), 
                'max':round_sig(np.max(elapsed_times),3), 
                'min':round_sig(np.min(elapsed_times), 3), 
        }
        
        output_dict = {
                'trials':trials, 
                'metric': time_type,
                **stat_dict
                }

        if return_results:
            output_dict['results'] = results


        return output_dict            

    @property
    def splits(self):
        return list(self.dataset.keys())


    @property
    def config_name(self):
        return self.config['dataset']['name']
    
    @property
    def raw_info(self):
        raw_info = self.list_configs(path=self.path)[self.config_name]
        # st.write(raw_info)
        return raw_info
    subtype = flavor = config_name



    @property
    def info(self):
        info = deepcopy(self.dataset[self.splits[0]].info.__dict__)
        assert 'splits' in info
        split_info = {}
        split_file_info = self.get_split_file_info()
        for split in self.splits:
            assert split in split_file_info
            split_info[split] = info['splits'][split].__dict__
            split_info[split]['file_info'] = split_file_info[split]
        
        
        info['splits'] = split_info    

        def json_compatible( x):
            try:
                json.dumps(x)
                return True

            except TypeError as e:
                return False   
        
        feature_info = {}
        def filter_features(input_dict):
            if isinstance(input_dict, dict):
                keys = list(input_dict.keys())
            elif isinstance(input_dict, list):
                keys = list(range(len(input_dict)))
            else:
                return input_dict
            for k in keys:
                v = input_dict[k]
                if not json_compatible(v):
                    if isinstance(v,pyarrow.lib.DataType):
                        input_dict[k] = repr(v)
                    elif type(v) in [dict, list, tuple, set]:
                        continue
                    else:
                        if hasattr(v, '__dict__'):
                            input_dict[k] = v.__dict__
                        elif hasattr(v, '__repr__'):
                            input_dict[k] =  v.__dict__
                        else:
                            raise NotImplementedError(v)
                            

            return  input_dict

        
       
        info['features'] = dict_fn(info['features'], filter_features)
        info['task_templates'] = dict_fn({'obj': info['task_templates']}, filter_features)['obj']
        info['task_templates'] = []
        
        info['version'] = str(info['version'])
        return info


    @property
    def configs(self):
        configs = [self.config_name]
        for service in self.services:
            if service.type == 'access':
                configs.append(service.additional_information['info']['config_name'])
        
        
        return list(set(configs))

    def add_service(self):
        asset = self.asset
        service = self.create_service()
        asset.add_service(service)
        self.algocean.ocean.asset.update(asset)

    @property
    def id(self):
        return f"{self.path}/{self.config_name}"
    
    @staticmethod
    def shard( dataset, shards=10, return_type='list'):
        '''
        return type options are list or dict
        '''


        if return_type == 'list':
            return [dataset.shard(shards, s) for s in range(shards)]
        elif return_type == 'dict':
             return DatasetDict({f'shard_{s}':dataset.shard(shards, s) for s in range(shards)})

        else:
            raise NotImplemented




    def strealit_sidebar(self):

        # self.load_state(**self.config.get('dataset'))
        

        with st.sidebar.form('Get Dataset'):


            dataset = st.selectbox('Select a Dataset', self.list_datasets(),0)
            self.load_builder(dataset)    

            config_dict = self.list_configs(dataset, return_type='dict')
            config_name = st.selectbox('Select a Config', list(config_dict.keys()), 0)

            config = config_dict[config_name]
            dataset_config = dict(path=dataset, name=config)

            submitted = st.form_submit_button("load")

            if submitted:
                self.config['dataset'] = dict(path=dataset, name=config_name, split=['train'])
                # st.write(self.config['dataset'])
                self.load_state(**self.config['dataset'])
                self.create_asset()



    def streamlit(self):
        self.strealit_sidebar()

    _tags = {}
    @property
    def tags(self):
        tags = self.get_tags(dataset=self.path, return_type='dict')
        return {**tags, **self._tags}
    
    def remove_tags(self, tags:list):
        for tag in tags:
            assert isinstance(tag,str), f'{tag} is not valid fam'
            self._tags.pop(tag)

    
    def add_tags(self, tags:dict):
        self._tags.update(tags)
        


    def get_tags(self, dataset:str='wikitext', return_type='dict'):
        if dataset == None:
            dataset = self.path
        rows = self.list_datasets(filter_fn=lambda r: r['id'] ==  dataset)

        rows = rows[rows['id'] == dataset]

        if len(rows) == 0:
            tags = {}
        else:
            assert len(rows) == 1, len(rows)
            tags = rows.iloc[0]['tags']

        if return_type == 'dict':
            return tags
        elif return_type == 'list':
            return [f'{k}:{v}' for k,v in tags.items()]
        else:
            raise NotImplementedError(f"return_type:{return_type} is not supported")
        


    def get_task(self, dataset:str='wikitext'):
        if dataset == None:
            dataset = self.path
        rows = self.list_datasets(filter_fn=f' r["id"] ==  "{dataset}"')
        assert len(rows) == 1, len(rows)
        return rows.iloc[0]['task_categories']
    
    @staticmethod
    def demo():
        module = DatasetModule(override={'load_dataset': False})

        df = module.list_datasets(filter_fn = 'r["tags"].get("size_categories") == "10K<n<100K"')
        module.set_default_wallet('richard')
        # st.write(df)
        # st.write(module.asset.__dict__)
        # st.write(module.create_asset(force_create=False).__dict__)
        # st.write(module.asset.services[0].__dict__)
        # module.download(destination='arshy')
        did = 'did:op:8d1f0eac0d8de08bb32eb6b9017c683a9f39bd1bb4e2914f80ff4805d01e3a11'
        asset = module.algocean.get_asset(f'{did}')
        module.algocean.download_asset(asset, destination='mnist')
        # dataset_list = list(df['id'][:20])
        # for dataset in dataset_list:
        #     override = {'dataset': {"path":dataset, "split":["train"], "load_dataset": True}}
        #     try:
        #         module = DatasetModule(override=override,)
        #         st.write(module.create_asset(force_create=False).__dict__)
        #     except ImportError as e:
        #         st.write('IMPORT ERROR', dataset)


if __name__ == '__main__':
    import streamlit as st
    import numpy as np
    from algocean.utils import *
    DatasetModule.demo()



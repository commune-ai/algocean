import hub
import os, sys
sys.path.append(os.environ['PWD'])

from typing import List,Optional
from dataclasses import dataclass,field
import json
from pathlib import Path
import numpy as np
from algocean import BaseModule
import datetime
import hub
from copy import deepcopy
@dataclass
class ActiveLoopModule(BaseModule):
    src:str='hub://activeloop'
    path:Optional[str]=None
    default_config_path = 'activeloop'
    default_splits = ['train']
    default_token_name = 'token'

    def __init__(self, config=None, override={}, **kwargs):
        BaseModule.__init__(self, config=config,override=override)
        self.algocean = self.get_object('ocean.OceanModule')(override={'network':self.config.get('network')})
        self.hub = hub
        self.load_state()

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
    def set_network(self, **kwargs):
        return self.algocean.set_network(**kwargs)
        

    @property
    def dataset_name(self):
        return self.path

    def split_info(self, split='train'):
        return self.split_info[split]
 
    @property
    def splits_info(self):
        split_info = {split:{} for split in self.splits}
        for split in self.splits:
            split_info[split]['name']=split
            split_info[split]['samples']=len(self.dataset[split])
        
        return split_info


    def load_state(self):
        self.splits = self.config.get('splits',self.default_splits)
        self.api_key = self.config.get('api_key')
        self.src = self.config.get('src', self.src)
        self.path = self.config.get('path')

        self.load_dataset()
        # self.config['metadata'] = self.metadata
        self.config['info'] = self.info
        self.config['src'] = self.src

    def load_dataset(self):

        self.dataset = {}
        for split in self.splits:
            split_ds = "-".join([self.path,split])
            url = "/".join([self.src,split_ds])
            ds = self.hub.load(url,token=self.api_key)
            self.dataset[split] = ds

            

    
    def automatic_ds(self):

        return hub.ingest(self.src, self.path)

    def get_dataset_metadata(self,project_name,parse=False):

        metadata_dict = {}

        parent = Path(project_name)

        files = [f for f in parent.iterdir() if not str(f.name).startswith("_")]

        for f in files:
            if os.path.isdir(str(f)):
                for j in f.glob("*.json"):
                    with open(j) as metadata:
                        json_file = json.loads(metadata.read())

                    metadata_dict.update({metadata.name:json_file})

            else:
                with open(f) as metadata:
                    json_file = json.loads(metadata.read())

                    metadata_dict.update({metadata.name:json_file})


        if parse:
            metadata_dict = self._parse_metadata(metadata_dict)
            return metadata_dict


        return metadata_dict


    def create_tensors(self,labels:list,htype:list,compression:list,class_names:list):

        for label,htype,compression,class_names in zip(labels,htype,compression,class_names):
            ds.create_tensor(label,htype=htype, sample_compression =compression)

    def populate(self, split='train', file_list =None):
        ds = self.dataset[split]
        with ds:
            # Iterate through the files and append to hub dataset
            for file in files_list:
                label_text = os.path.basename(os.path.dirname(file))
                label_num = class_names.index(label_text)

                #Append data to the tensors
                ds.append({'images': hub.read(file), 'labels': np.uint32(label_num)})


    @hub.compute
    @staticmethod
    def filter_labels(sample_in, labels_list):
        return sample_in.labels.data()['text'][0] in labels_list

    @staticmethod
    def max_array_length(arr_max, arr_to_compare):  # helper for __str__
        for i in range(len(arr_max)):
            str_length = len(arr_to_compare[i])
            if arr_max[i] < str_length:
                arr_max[i] = str_length
        return arr_max

    @property
    def info(self):
        info = {**self.raw_info}
        info['features'] = self.features_info
        info['splits'] = self.splits_info
        info['description'] = info.get('description', 'No Description Available')
        return info

    @property
    def feature_names(self):
        return list(self.features_info.keys())

    @property
    def features_info(self):

        metadata = {}
        dataset = self.dataset[self.splits[0]]
        tensor_dict = dataset.tensors

        for tensor_name in tensor_dict:

            metadata[tensor_name] = {}

            tensor_object = tensor_dict[tensor_name]

            tensor_htype = tensor_object.htype
            if tensor_htype == None:
                tensor_htype = "None"

            tensor_compression = tensor_object.meta.sample_compression
            if tensor_compression == None:
                tensor_compression = "None"

            tensor_dtype = tensor_object.dtype
            if  tensor_dtype == None:
                tensor_dtype = "None"

            #Working - Improvement to resolve - ValueError: dictionary update sequence element #0 has length 10; 2 is required
            if tensor_name == "labels":
                label_distr = np.unique(tensor_object.data()["value"],return_counts=True)
                metadata[tensor_name].update({"label_distribution": dict(zip(label_distr[0].tolist(), label_distr[1].tolist()))})

            shape = tensor_object[:1].shape
            tensor_shape = (tensor_object.shape_interval if None in shape else shape)[1:]

            metadata[tensor_name].update({"htype":tensor_htype})
            metadata[tensor_name].update({"shape":tensor_shape})
            metadata[tensor_name].update({"compression":tensor_compression})
            metadata[tensor_name].update({"dtype":str(tensor_dtype)})

        return metadata


        
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



    def save(self,mode:str='estuary', replace=False):

        if mode == 'estuary':
            state_path_map = {}
            for split, dataset in self.dataset.items():
   
                self.save_to_disk(splits=[split], replace=replace)
                split_path = os.path.join(self.local_tmp_dir, split)
                split_state = self.client.estuary.add(path=split_path)

                state_path_map[split] = {f.replace(split_path+'/',''): cid for f,cid in split_state.items()}

        elif mode == 'ipfs':
            state_path_map = {}
            for split, dataset in self.dataset.items():
                cid = self.client.ipfs.save_dataset(dataset)
                state_path_map[split] = self.client.ipfs.info(cid)

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
                path = self.client.ipfs.load_dataset(cid, mode='activeloop')
            self.dataset = datasets.load_from_disk(path=path)
        else:
            raise NotImplementedError

        return self.dataset

    @property
    def raw_info(self):
        ds = self.dataset[self.splits[0]]
        if isinstance(ds._info, dict):
            return ds._info
        else:
            return self.dataset[self.splits[0]]._info.__dict__['_info']
    @property
    def citation(self):
        return self.raw_info['citation']

    @property
    def description(self):
        return self.raw_info['description']

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

    def hash(self, *args, **kwargs):
        return self.algocean.hash(*args, **kwargs)

    @property
    def split_file_info(self):

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
                    file_hash = self.hash(cid))
                )



        # url_files = 

        return split_url_files_info
    

    def additional_information(self, mode='service'):

        info_dict = {
            'organization': 'activeloop',
            'package': {
                        'name': 'hub',
                        'version': str(self.hub.__version__)
                        },
            # 'info': {'configs': self.configs}, 
        }

        if mode == 'service':
            info_dict['info'] = self.info
            for split in self.splits:
                info_dict['info']['splits'][split]['file_info'] = self.split_file_info[split]

        elif mode == 'service.access':
            info_dict['info'] = self.info
        elif mode == 'service.compute':
            info_dict['info'] = self.info  
        elif mode == 'asset':
            info_dict['info'] = self.info
        else:
            raise NotImplementedError(f'{mode} not supported')

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
                        force_create=True,
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
            name = 'access'
        if files == None:
            files = self.url_files
        name = '.'.join([name, service_type])

        service = self.get_service(name)
        if service != None and force_create==False:
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

        metadata['name'] = self.config.get('path')
        metadata['description'] = self.info['description']
        metadata['author'] = self.algocean.wallet.address
        metadata['license'] = self.info.get('license', "CC0: PublicDomain")
        metadata['categories'] = self.info.get('categories', [])
        metadata['tags'] = [f'{k}:{v}' for k,v in self.tags.items()]
        metadata['additionalInformation'] = self.additional_information('asset')
        metadata['type'] = 'dataset'
        current_datetime = datetime.datetime.now().isoformat()
        metadata["created"]=  current_datetime
        metadata["updated"] = current_datetime

        return metadata

    @property
    def features(self):
        return self.info['features']

    @property
    def datatokens(self):
        return self.get_datatokens(asset=self.asset)

    @property
    def wallet(self):
        return self.algocean.wallet

    @property
    def wallets(self):
        return self.algocean.wallets

    def set_default_wallet(self, *args, **kwargs):
        return self.algocean.set_default_wallet(*args, **kwargs)

    @property
    def default_wallet_key(self):
        return self.algocean.default_wallet_key

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
        return self.assets2info(self.my_assets)

    @staticmethod
    def assets2info(assets:list): 
        asset_info_list = []
        for a in assets:
            get_map = {
                'name': 'metadata.name',
                'organization': 'metadata.additionalInformation.organization',
                'did': 'did',
                'chain_id': 'chain_id'
            }
            asset_info_list.append({k: dict_get(a.__dict__, v) for k,v in get_map.items()})
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


        
    def download(self, service=None, destination='fam'):
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
 
    @property
    def splits(self):
        return self._splits

    def resolve_var(self, **kwargs):
        assert len(kwargs) == 0
        for k,v in kwargs.items():
            if v == None:
                return getattr(self,k)
            else:
                return v
            
        raise Exception(f'{kwargs}, should only have one key')

    @splits.setter
    def splits(self, value:list):
        self._splits = value
    
    
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
    def add_service(self):
        asset = self.asset
        service = self.create_service()
        asset.add_service(service)
        self.algocean.ocean.asset.update(asset)
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

    
    @staticmethod
    def shard( dataset, shards=10, return_type='list'):
        raise NotImplementedError



    _tags = {}
    @property
    def tags(self):
        tags = self.config.get('tags', [])
        
        if isinstance(tags, list):
            if len(tags) == 0:
                tags = {}
            elif isinstance(tags[0], str):
                tags = {t.split(':')[0]: t.split(':')[1] for t in tags}
        else:
            raise NotImplementedError

        return {**tags, **self._tags}
    
    def remove_tags(self, tags:list):
        for tag in tags:
            assert isinstance(tag,str), f'{tag} is not valid fam'
            self._tags.pop(tag)

    
    def add_tags(self, tags:dict):
        self._tags.update(tags)
        
    def get_task(self, dataset:str='wikitext'):
        raise NotImplementedError

    @staticmethod
    def dataset_splits(dataset):
        datasets = ActiveLoopModule.datasets(return_type='dict')
        
        return datasets[dataset]['splits']

    @staticmethod
    def datasets(return_type='dict'):

        ds_list =  [ds for ds in hub.list('activeloop')]
        ds_split_dict = {}

        for ds in ds_list:
            if len( ds.split('-')) != 2:
                continue
            ds, split = ds.split('-')
            ds = ds.replace('activeloop/', '')
            if ds not in ds_split_dict:
                ds_split_dict[ds] = {'splits': [split]}
            else:
                ds_split_dict[ds]['splits'].append(split)

        if return_type== 'list':
            return list(ds_split_dict.keys())
        elif return_type== 'dict': 
            return ds_split_dict
        else:
            raise NotImplementedError

    __file__ = __file__
    @property
    def local_tmp_dir(self):
        return f'/tmp/{self.module_path()}/{self.path}'

    def load_from_disk_config(self, path):
        path = os.path.join(path, 'config.json')
        config = self.client.local.get_json(path=path )
        return config


    def load_from_disk(self, path=None, splits=None, chunk_limit=2):
        


        config = self.client.local.get_json(os.path.join(path, 'config.json') )
        dataset = {}


        if isinstance(splits, type(None)):
            splits = self.splits
        elif isinstance(splits, str):
            assert splits in self.splits
        elif isinstance(splits, list):
            pass


        if path  == None:
            path = self.local_tmp_dir


        for split in splits:

            split_path = os.path.join(path, split)

            ds = hub.empty(split_path, overwrite=True, token = config['api_key'])

            for files in self.client.local.glob(split_path+'/**'):
                feature_config_dict = config['info']['features']
                for feature, feature_config in feature_config_dict.items():
                    if feature_config['compression'] == 'None':
                        feature_config['compression'] = None

                    ds.create_tensor(feature, htype=feature_config['htype'], sample_compression=feature_config['compression'])
                    feature_files = sorted([f for f in files if f.startswith(f'{split}-{feature}')])
                    for f in feature_files[:chunk_limit]:
                        tensor = np.load(f)
                        getattr(ds, feature).extend(tensor) 

            ds._info = config.get('info', {})
            dataset[split] = ds
            # for file in files:
            #     if file.startswith('config'):
            #         slef

        return dataset

        # for split in self.splits:
        #     split_ds = "-".join([self.path,split])
        #     url = "/".join([self.src,split_ds])

        #     self.dataset[split] = self.hub.load(url,token=self.api_key)




    

    def save_to_disk(self, path=None, splits=None, chunksize=1000, max_samples=1000, replace=True):


        chunks = max_samples // chunksize
        dir_path  = self.local_tmp_dir
        self.client.local.makedirs(dir_path, True)

        
        if isinstance(splits, type(None)):
            splits = self.splits
        elif isinstance(splits, str):
            assert split in self.splits
            splis = [split]
        elif isinstance(splits, list):
            pass

        
        if path  == None:
            path = self.local_tmp_dir

        for split in splits:


            split_path = os.path.join(path, split)

            if replace:
                if self.client.local.isdir(split_path):
                    self.client.local.rm(split_path, recursive=True)
                self.client.local.makedirs(split_path, True)
            else:
                if self.client.local.exists(split_path):
                    continue 
                
            self.client.local.put_json(path=os.path.join(split_path,'config.json'), data=self.config)

            dataset = self.dataset[split][:max_samples]
            for feature, feature_tensor in dataset.tensors.items():
                feature_tensor_chunks = np.array_split(feature_tensor, chunks, axis=0)
                for chunk_id, feature_tensor_chunk in enumerate(feature_tensor_chunks):
                    feature_path = os.path.join(split_path,f"{feature}-{chunk_id}.npy")
                    np.save(feature_path, feature_tensor_chunk, allow_pickle=False, fix_imports=True)


    @staticmethod
    def streamlit():
        ActiveLoopModule.demo()
        
    @staticmethod
    def demo():
        datasets_dict = ActiveLoopModule.datasets('dict')
        dataset_options = list(datasets_dict.keys())
        ds2index = {ds:i for i,ds in enumerate(dataset_options)}
        ds = st.sidebar.selectbox('Select a Dataset',dataset_options, ds2index['mnist'] )
        ds_splits = datasets_dict[ds]['splits']
        ds_splits = st.sidebar.multiselect('Select Splits', ds_splits, ds_splits)
        

        error_datasets =  ['kmnist', 'imagenet', 'coco', 'fashionpedia', 'hockey', 'davis2017', 'gtzan', 'daisee', 'kth', 'gtsrb', 'timit', 'mura', 'svhn', 'lincolnbeet']
        demo_datasets =  ['hasy', 'cifar100', 'k49', 'mnist', 'stl10', 'cifar10', 'stanford']

        filtered_datasets = []
        for ds in demo_datasets:

            st.write('DATASET: ',ds)
            ds_splits = datasets_dict[ds]['splits']
            module = ActiveLoopModule(override={'path': ds,  'splits': ds_splits, 'network': 'local'})
            st.write(module.create_asset(force_create=False))

    # def load_from_disk(self):
    #     config = self.load_from_disk_config()
    #     for split in config['splits']:

if __name__ == '__main__':
    import streamlit as st
    # from metadata import DatasetMetdata
    import os
    from algocean.utils import *


    module = ActiveLoopModule(override={'path': 'mnist',  'splits': ['train'], 'network': 'mumbai'})
    module.set_default_wallet('richard')

    # st.write(module.asset.__dict__)
    # split = 'train'
    st.write()
    # st.write(module.download())
    

    st.write(module.my_assets_info)
    # st.write([a.__dict__ for a in module.my_assets])
    # st.write(module.asset)




    
        
    
    
    st.write()

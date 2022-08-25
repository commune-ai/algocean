
import os, sys
sys.path.append(os.environ['PWD'])
import datasets 
import datetime
import transformers
from copy import deepcopy
from typing import Union
from copy import deepcopy
from algocean import BaseModule
import torch
import ray
from algocean.utils import dict_put
from datasets.utils.py_utils import asdict, unique_values

import fsspec
import os
from ipfsspec.asyn import AsyncIPFSFileSystem
from fsspec import register_implementation
import asyncio
import io

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, load_dataset_builder




# Create Ocean instance
import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))

from algocean.utils import RecursiveNamespace, dict_put, dict_has
import datetime
from ocean_lib.assets.asset import Asset
from ocean_lib.example_config import ExampleConfig
from ocean_lib.web3_internal.contract_base import ContractBase
from ocean_lib.models.datatoken import Datatoken
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.structures.file_objects import IpfsFile, UrlFile
from ocean_lib.services.service import Service
from ocean_lib.structures.file_objects import FilesTypeFactory
from ocean_lib.exceptions import AquariusError

from typing import *
# Create Alice's wallet

from ocean_lib.config import Config
from ocean_lib.models.data_nft import DataNFT
from ocean_lib.web3_internal.wallet import Wallet
from ocean_lib.web3_internal.constants import ZERO_ADDRESS


# from web3._utils.datatypes import Contract

import fsspec

from ocean_lib.structures.file_objects import UrlFile
from algocean import BaseModule
# from algocean import BaseModule


class OceanModule(BaseModule):
    default_cfg_path = 'ocean.module'
    default_wallet_key = 'default'
    wallets = {}
    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)

        self.initialize_state()
        if 'ocean' in self.config:
            self.set_ocean(ocean_config=self.config.get('ocean'))
        if 'network' in self.config:
            self.set_network(network=self.config.get('network'))

        self.load_wallets(self.config.get('wallet'))

    
    def initialize_state(self):
        # self.datanfts = {}
        # self.datatokens = {}
        # self.dataassets = {}
        pass


    def load_wallets(self, wallet_config:dict):
        '''
        Load Private Key variable into your
         wallet when pointing to the private 
         key env.

         wallet:
            alice: PRIVATE_KEY1
            bob: PRIVATE_KEY2


        or you can put in the keys manually
         but thats too deep
        '''

        for k,pk in wallet_config.items():
            assert isinstance(pk, str)
            self.add_wallet(wallet_key=k, private_key=pk)

    @property
    def network(self):
        return self._network
    def set_network(self, network:str=None):
        '''
        set the network
        defaults to local fork
        '''
        if network == None:
            network = 'local'
        self._network = network
        self.set_ocean(ocean_config=f'./config/{network}.in')

        

    
    def set_ocean(self, ocean_config):
        self.config['ocean'] = self.get_ocean(ocean_config, return_ocean=False)
        self.ocean = Ocean(self.config['ocean'])
        self.web3 = self.ocean.web3
        self.aquarius = self.ocean.assets._aquarius
        


    
    @staticmethod
    def get_ocean( ocean_config, return_ocean=True):
        if ocean_config == None:
            ocean_config =  ExampleConfig.get_config()
        elif isinstance(ocean_config, str):
            if ocean_config.startswith('./'):
            
                ocean_config = os.path.dirname(__file__) + ocean_config[1:]
            ocean_config = Config(filename=ocean_config)
        
        elif isinstance(ocean_config, Config):
            ocean_config = ocean_config
        else:
            raise NotImplementedError  
        
        assert isinstance(ocean_config, Config), 'ocean_config must be type Config'
        if return_ocean:
            return Ocean(ocean_config)
        return ocean_config



    def get_existing_wallet_key(self, private_key:str=None, address:str=None):
        for w_k, w in self.wallets.items():
            if private_key==w.private_key or address == w.address:
                return w_k

        return None

    def add_wallet(self, wallet_key:str='default', private_key:str='TEST_PRIVATE_KEY1', wallet:Wallet=None):
        '''
        wallet_key: what is the key you want to store the wallet in
        private_key: the key itself or an env variable name pointing to that key
        '''
        if isinstance(wallet,Wallet):
            self.wallets[wallet_key] = wallet
            return wallet
        # fetch the name or the key
        private_key = os.getenv(private_key, private_key)

        existing_wallet_key = self.get_existing_wallet_key(private_key=private_key)
        # if the key is registered, then we will swtich the old key with the new key
        if existing_wallet_key == None:
            self.wallets[wallet_key] = self.generate_wallet(private_key=private_key)
        else:
            self.wallets[wallet_key] =  self.wallets.pop(existing_wallet_key)
        self.ensure_default_wallet()
        return self.wallets[wallet_key]

    def generate_wallet(self, private_key:str):
        private_key = os.getenv(private_key, private_key)
        return Wallet(web3=self.web3, 
                      private_key=private_key, 
                      block_confirmations=self.config['ocean'].block_confirmations, 
                      transaction_timeout=self.config['ocean'].transaction_timeout)  
    
    def rm_wallet(self, key):
        '''
        remove wallet and all data relating to it
        '''
        del self.wallets[key]
        self.ensure_default_wallet()

    def list_wallets(self, return_keys=True):
        '''
        list wallets
        '''
        if return_keys:
            return list(self.wallets.keys())
        else:
            return  [(k,v) for k,v in self.wallets.items()]

    @property
    def wallet(self):
        # gets the default wallet
        return self.wallets[self.default_wallet_key]

    def set_default_wallet(self, key:str):
        self.default_wallet_key = key
        return self.wallets[self.default_wallet_key]

    def ensure_default_wallet(self):
        if self.default_wallet_key not in self.wallets:
            if len(self.wallets) > 0:
                self.default_wallet_key = list(self.wallets.keys())[0]

    def get_wallet(self, wallet, return_address=False):
        if wallet == None:
            wallet = self.wallet
        elif isinstance(wallet, str):
            if self.web3.isAddress(wallet):
                assert return_address
                return wallet
            
            wallet = self.wallets[wallet]
        elif isinstance(wallet,Wallet):
            wallet = wallet
        else:
            raise Exception(f'Bro, the wallet {wallet} does not exist or is not supported')

        assert isinstance(wallet, Wallet), f'wallet is not of type Wallet but  is {Wallet}'
    
        if return_address:
            return wallet.address
        else: 
            return wallet


    def create_datanft(self, name:str , symbol:str=None, wallet:Union[str, Wallet]=None):
        wallet = self.get_wallet(wallet=wallet)

        if symbol == None:
            symbol = name
        nft_key =  symbol
        datanft = self.get_datanft(datanft=nft_key, handle_error=True)
        if datanft == None:
            datanft = self.ocean.create_data_nft(name=name, symbol=symbol, from_wallet=wallet)

        return datanft

    def list_datanfts(self):
        return list(self.datanfts.keys())


    def generate_datatoken_name(self, datanft:Union[str, DataNFT]=None):
        datanft = self.get_datanft(datanft)
        index =  0 
        nft_token_map =  self.nft_token_map(datanft)
        while True:
            token_name = f'DT{index}'
            if token_name not in nft_token_map:
                return token_name
    def create_datatoken(self, name:str, symbol:str=None, datanft:Union[str, DataNFT]=None, wallet:Union[str, Wallet]=None):
        wallet = self.get_wallet(wallet)
        datanft = self.get_datanft(datanft)
        datatokens_map = self.get_datatokens(datanft=datanft, return_type='map')
        datatoken = datatokens_map.get(name)
        
        symbol = symbol if symbol != None else name
        if datatoken == None:
            datatoken = datanft.create_datatoken(name=name, symbol=symbol, from_wallet=wallet)

        assert isinstance(datatoken, Datatoken), f'{datatoken}'
        return datatoken



    def get_contract(self, address:str, contract_class=ContractBase):
        return contract_class(web3=self.web3, address=address)
    
    def get_address(self, contract):
        return contract.address


    # @staticmethod
    # def get_asset_did(asset:Asset):
    #     return asset.did

    def get_assets(self, wallet:Union[str, Wallet]=None, return_type='dict'):
        '''
        get assets from wallet
        '''
        
        wallet = self.get_wallet(wallet)
        text_query = f'metadata.author:{wallet.address}' 
        # current_time_iso = datetime.datetime.now().isoformat()
        assets = self.search(text=text_query, return_type='asset')
        return assets

    def get_asset(self, datanft=None, did=None, handle_error=False, timeout=10):
        '''
        get asset from datanft using aquarius
        '''

        if isinstance(datanft, Asset):
            return datanft

        datanft =self.get_datanft(datanft)
        query_text = f'nft.address:{datanft.address}'
            
        timer = Timer(start=True)

        while timer.elapsed_time() < timeout:
            try:
                time.sleep(0.1)
                st.write(timer.elapsed_time())
                assets = self.search(text=query_text, return_type='asset')
                assert len(assets)==1, f'This asset from datanft: {datanft.address} does not exist'
                assert isinstance(assets[0], Asset), f'The asset is suppose to be an Asset My guy'
                return assets[0]
            except Exception as e:
                if handle_error:
                    return None
                else:
                    raise(e)
        



    def get_wallet_datanfts(self, wallet:Union[str, Wallet]=None):
        wallet_address = self.get_wallet(wallet, return_address=True)
        return self.search(text='metadata.address:{wallet_address}', return_type='asset')
        


    # def assets(self, datanft, wallet):


    
    @staticmethod
    def fill_default_kwargs(default_kwargs, kwargs):
        return {**default_kwargs, **kwargs}


    def nft_token_map(self, datanft=None, return_type='map'):
        '''
        params:
            return_type: the type of retun, options
                options:
                    - map: full map (dict)
                    - key: list of keys (list[str])
                    - value: list of values (list[Datatoken])
        
        '''
        supported_return_types = ['map', 'key', 'value']
        assert return_type in supported_return_types, \
              f'Please specify a return_type as one of the following {supported_return_types}'

        
        datanft =self.get_datanft(datanft)
        datanft_symbol = datanft.symbol()
        output_token_map = {}

        for k,v in self.datatokens.items():
            k_nft,k_token = k.split('.')
            if datanft_symbol == k_nft:
                output_token_map[k_token] = v


        if return_type in ['key']:
            return list(output_token_map.keys())
        elif return_type in ['value']:
            return list(output_token_map.values())
        elif return_type in ['map']:
            return output_token_map

        raise Exception('This should not run fam')


    def dispense_tokens(self, 
                        datatoken:Union[str, Datatoken]=None, 
                        datanft:Union[str, DataNFT]=None,
                        amount:int=100,
                        destination:str=None,
                        wallet:Union[str,Wallet]=None):
        wallet = self.get_wallet(wallet)
        amount = self.ocean.to_wei(amount)
        datatoken = self.get_datatoken(datatoken=datatoken, datanft=datanft)
        if destination == None:
            destination = wallet.address 
        else:
            destination = self.get_wallet(destination, return_address=True)


        # ContractBase.to_checksum_address(destination)


        self.ocean.dispenser.dispense(datatoken=datatoken.address,
                                     amount=amount, 
                                    destination= destination,
                                     from_wallet=wallet)

    def create_dispenser(self,
                        datatoken:Union[str, Datatoken]=None, 
                        datanft:Union[str, DataNFT]=None,
                        max_tokens:int=100, 
                        max_balance:int=None, 
                        with_mint=True, 
                        wallet=None,
                        **kwargs):

        datatoken=  self.get_datatoken(datatoken=datatoken, datanft=datanft)
        wallet = self.get_wallet(wallet)

        dispenser = self.ocean.dispenser

        max_tokens = self.ocean.to_wei(max_tokens)
        if max_balance == None:
            max_balance = max_tokens
        else:
            max_balance = self.ocean.to_wei(max_tokens)
        # Create dispenser

        datatoken.create_dispenser(
            dispenser_address=dispenser.address,
            max_balance=max_balance,
            max_tokens=max_tokens,
            with_mint=with_mint,
            allowed_swapper=ZERO_ADDRESS,
            from_wallet=wallet,
        )

    

    def get_datanft(self, datanft:Union[str, DataNFT]=None, handle_error:bool= False):
        '''
        dataNFT can be address, key in self.datanfts or a DataNFT
        '''

        try:

            if isinstance(datanft, DataNFT):
                datanft =  datanft

            elif isinstance(datanft, str):
                if self.web3.isAddress(datanft):
                    datanft = DataNFT(address=datanft)
                elif datanft in self.datanfts :
                    datanft = self.datanfts[datanft]
                else:
                    raise NotImplementedError(f'{datanft} is not found')
            
            assert isinstance(datanft, DataNFT), f'datanft should be in the formate of DataNFT, not {datanft}'
            return datanft
        except Exception as e:
            if handle_error:
                return None
            else:
                raise(e)



    def create_asset(self,datanft, services:list, metadata:dict=None, wallet=None, **kwargs ):
        wallet = self.get_wallet(wallet)
        datanft = self.get_datanft(datanft=datanft)
        asset = self.get_asset(datanft, handle_error=True)
        if asset != None:
            assert isinstance(asset, Asset)
            return asset
               
        if metadata == None:
            metadata = self.create_metadata(datanft=datanft,wallet=wallet, **kwargs.get('metadata', {}))
        
        default_kwargs= dict(
        data_nft_address = datanft.address,
        deployed_datatokens = [self.get_datatoken(s.datatoken) for s in services],
        publisher_wallet= wallet,
        metadata= metadata,
        services=services
        )

        kwargs = {**kwargs, **default_kwargs}

        
        if asset == None:
            asset = self.ocean.assets.create(**kwargs)
        
        return asset

            
    

    def dummy_files(self, mode='ipfs'):
        cid = self.client.ipfs.put_json(data={'dummy':True})
        return self.create_files([{'hash':f'{cid}', 'type':'ipfs'}]*1)

    @staticmethod
    def create_files(file_objects:Union[list, dict]=None, handle_null=False):
        if isinstance(file_objects, dict):
            file_objects = [file_objects]
        assert isinstance(file_objects, list) 
        assert isinstance(file_objects[0], dict)

        output_files = []
        for file_object in file_objects:
            output_files.append(FilesTypeFactory(file_object))

        return output_files



    def mint(self, to:Union[str,Wallet], value:int=1,datanft:str=None, datatoken:str=None, wallet:Wallet=None , encode_value=True):
        wallet = self.get_wallet(wallet=wallet)
        to_address = self.get_wallet(wallet=to, return_address=True)
        datatoken = self.get_datatoken(datanft=datanft,datatoken=datatoken)
        
        if encode_value:
            value = self.ocean.to_wei(str(value))
        
        assert datatoken != None, f'datatoken is None my guy, args: {dict(datanft=datanft, datatoken=datatoken)}'
        datatoken.mint(account_address=to_address, 
                        value=value, from_wallet=wallet )


    def get_datatoken(self, datatoken:str=None, datanft:str=None) -> Datatoken:

        if isinstance(datatoken, Datatoken):
            return datatoken

        if isinstance(datatoken, str):
            if self.web3.isAddress(datatoken): 
                return Datatoken(web3=self.web3,address=datatoken)


            datatokens_map = self.get_datatokens(datanft=datanft, return_type='map')
            if datatoken in datatokens_map:
                return datatokens_map[datatoken]
        else:
            raise Exception(f'BRO {datanft} is not define')

    
    # @property
    # def datatokens(self):
    #     return {}

    def get_balance(self,wallet:Union[Wallet,str]=None, datanft:str=None, datatoken:str=None):
        
        wallet_address = self.get_wallet(wallet=wallet, return_address=True)
        datatoken = self.get_datatoken(datanft=datanft, datatoken=datatoken )
        if datatoken == None:
            value =  self.web3.eth.get_balance(wallet_address)
        else:
            value =  datatoken.balanceOf(wallet_address)
        
        return value
   
    @property
    def assets(self):
        return self.get_assets()

    @property
    def datatokens(self):
        dt_list = []
        for asset in self.assets:
            dt_list += self.get_datatokens(asset=asset)
        return [dt for dt in dt_list]

    @property
    def services(self):
        services = []
        for asset in self.get_assets():
            services += asset.services

        return services
    
    @property
    def datanfts(self):
        datanfts = []
        for asset in self.get_assets():
            datanfts += [DataNFT(web3=self.web3, address=asset.nft['address'])]

        return datanfts

    def get_services(self, asset, return_type = 'service'):

        asset = self.get_asset(asset)

        supported_types = ['service', 'dict']
        assert return_type in supported_types
        
        if return_type== 'service':
            return asset.services
        elif return_type == 'dict':
            return [s for s in asset.services.__dict__]


    def create_service(self,
                        name: str,
                        service_type:str = 'access',
                        files:list = None,
                        datanft:Optional[str]=None,
                        datatoken: Optional[str]=None,
                        additional_information: dict = {},
                        wallet=None,**kwargs):
        wallet = self.get_wallet(wallet)
        datanft = self.get_datanft(datanft=datanft)
        datatoken = datatoken if datatoken != None else name
        datatoken = self.get_datatoken(datanft=datanft, datatoken=datatoken)

        if files == None:
            files = self.dummy_files()

        service_dict = dict(
            name=name,
            service_type=service_type,
            service_id= kwargs.get('id', name),
            files=files,
            service_endpoint=kwargs.get('service_endpoint', self.ocean.config.provider_url),
            datatoken=datatoken.address,
            timeout=kwargs.get('timeout', 3600),
            description = 'Insert Description here',
            additional_information= additional_information,
        )

        service_dict = {**service_dict, **kwargs}

        return Service(**service_dict)




    def get_services(self, asset=None, datanft=None, return_type='service'):
        if asset != None:
            asset = self.get_asset(asset)
        elif datanft != None:
            asset = self.get_asset(datanft)
        else:
            raise NotImplementedError
        

        services = asset.services
        if return_type == 'dict':
            services = [ s.__dict__ for s in services]
        elif return_type == 'service':
            services = services
        else:
            raise NotImplementedError
        
        return services





    def get_service(self, asset=None, service=None):
        if isinstance(service, Service):
            return service
        else:
            asset = self.get_asset(asset)
            if service == None:
                assert len(asset.services)>0, 'There are no services for the asset'
                return asset.services[0]
            elif isinstance(service, int):
                return asset.services[service]
            else:
                raise NotImplementedError
            
    def pay_for_access_service(self,
                              asset:Union[str,Asset],
                              service:Union[str,Service]=None,
                              wallet:Union[str, Wallet]=None, **kwargs):
        
        
        asset = self.get_asset(asset)
        service= self.get_service(asset=asset, service=service)
        wallet = self.get_wallet(wallet=wallet) 


        default_kwargs = dict(
            asset=asset,
            service=service,
            consume_market_order_fee_address=service.datatoken,
            consume_market_order_fee_token=wallet.address,
            consume_market_order_fee_amount=0,
            wallet=wallet,
        )

        kwargs = {**default_kwargs, **kwargs}

        order_tx_id = self.ocean.assets.pay_for_access_service( **kwargs )     

        return order_tx_id   
        

    def download_asset(self, asset, service=None, destination='./', order_tx_id=None,index=None, wallet=None):
        asset = self.get_asset(asset)
        service= self.get_service(asset=asset, service=service)
        wallet = self.get_wallet(wallet=wallet) 

        if order_tx_id == None:
            order_tx_id = self.pay_for_access_service(asset=asset, service=service, wallet=wallet)

        file_path = self.ocean.assets.download_asset(
                                        asset=asset,
                                        service=service,
                                        consumer_wallet=wallet,
                                        destination=destination,
                                        index=index,
                                        order_tx_id=order_tx_id
                                    )
        return file_path

    def create_metadata(self, datanft=None, wallet=None, **kwargs ):

        wallet = self.get_wallet(wallet)
        datanft = self.get_datanft(datanft)
        metadata ={}

        metadata['name'] = datanft.name
        metadata['description'] = kwargs.get('description', 'Insert Description')
        metadata['author'] = kwargs.get('author', wallet.address)
        metadata['license'] = kwargs.get('license', "CC0: PublicDomain")
        metadata['categories'] = kwargs.get('categories', [])
        metadata['tags'] = kwargs.get('tags', [])
        metadata['additionalInformation'] = kwargs.get('additionalInformation', {})
        metadata['type'] = kwargs.get('type', 'dataset')

        current_datetime = datetime.datetime.now().isoformat()
        metadata["created"]=  current_datetime
        metadata["updated"] = current_datetime

        return metadata



    def search(self, text: str, return_type:str='asset') -> list:
        """
        Search an asset in oceanDB using aquarius.
        :param text: String with the value that you are searching
        :return: List of assets that match with the query
        """
        # logger.info(f"Searching asset containing: {text}")

        ddo_list = [ddo_dict['_source'] for ddo_dict in self.aquarius.query_search({"query": {"query_string": {"query": text}}}) 
                        if "_source" in ddo_dict]
        
        if return_type == 'asset':
            ddo_list = [Asset.from_dict(ddo) for ddo in ddo_list]
        elif return_type == 'dict':
            pass
        else:
            raise NotImplementedError

        return ddo_list


    @staticmethod
    def describe(instance):
        supported_return_types = [ContractBase]
        
        if isinstance(instance, ContractBase):
            return instance.contract.functions._functions
        else:
            raise NotImplementedError(f'Can only describe {ContractBase}')

    def get_datatokens(self, datanft:Union[ContractBase]=None, asset=None, return_type:str='value'):


        datatokens = []
        if asset != None:
            asset =  self.get_asset(asset)
            datatoken_obj_list =  asset.datatokens
            for datatoken_obj in datatoken_obj_list:
                dt_address = datatoken_obj.get('address')
                datatokens += [Datatoken(web3=self.web3, address=dt_address)]
            
        elif datanft != None:
            datanft = self.get_datanft(datanft)   
            dt_address_list = datanft.contract.caller.getTokensList()
            datatokens = [Datatoken(web3=self.web3,address=dt_addr) for dt_addr in dt_address_list]
        
        else:
            raise NotImplementedError(f'datanft: {datanft}')

        supporrted_return_types = ['map', 'key', 'value']

        assert return_type in supporrted_return_types, f'{return_type} not in {supporrted_return_types}'
        if return_type in ['map']:
            return {t.symbol():t for t in datatokens }
        elif return_type in ['key']:
            return [t.symbol() for t in datatokens]
        elif return_type in ['value']:
            return datatokens
        else:
            raise NotImplementedError





class DatasetModule(BaseModule, Dataset):
    default_cfg_path = 'huggingface.dataset.module'

    datanft = None
    default_token_name='token' 

    
    dataset = {}
    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
        self.load_dataset_factory()
        self.load_dataset_builder()
        self.load_dataset()
        self.algocean = OceanModule()



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



    def load_dataset_factory(self, path=None):
        if path == None:
            path = self.config['dataset']['path']
        self.dataset_factory = datasets.load.dataset_module_factory(path)

    def load_dataset_builder(self):
        self.dataset_builder = datasets.load.import_main_class(self.dataset_factory.module_path)



    def load_dataset(self, *args, **kwargs):
        if len(args) + len(kwargs) == 0:
            config_kwargs = self.config.get('dataset')
            split = config_kwargs.get('split', ['train'])
            if isinstance(split, str):
                split = [split]
            if isinstance(split, list):
                split = {s:s for s in split}
            
            config_kwargs['split'] = split
            if isinstance(config_kwargs, list):
                args = config_kwargs
            elif isinstance(config_kwargs, dict):
                kwargs = config_kwargs

        self.dataset = load_dataset(*args, **kwargs)


        return self.dataset
    
    def get_split(self, split='train'):
        return self.dataset[split]


    @staticmethod
    def list_datasets():
        return datasets.list_datasets()

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
        
    @property
    def builder_configs(self):
        return {v.name:v for v in module.dataset_builder.BUILDER_CONFIGS}


    def save(self,mode:str='ipfs'):
        if mode == 'ipfs':
            path_split_map = {}
            for split, dataset in self.dataset.items():
                cid = self.client.ipfs.save_dataset(dataset)
                path_split_map[split] = self.client.ipfs.ls(cid)
            self.config['state_path'] = path_split_map
        else:
            raise NotImplementedError
    
        return self.config['state_path']




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
        return self.config['dataset']['path']

    @property
    def builder_name(self):
        return self.path

    @property
    def dataset_name(self):
        return self.path

    @property
    def config_name(self):
        return self.config['dataset']['name']

    def create_asset(self, datatoken='token', services:list=[] ):
        
        
        self.algocean.create_asset(datanft=self.dataset_name, datatoken=datatoken,
                                 services=services )

    @property
    def url_files_metadata(self):
        pass

    def get_url_data(self):
        split_path_map = self.save()

        file_index = 0
        url_files = []
        split_url_files_info = {}
        cid2index = {}
        for split, split_file_configs in split_path_map.items():
            split_url_files_info[split] = []
      
            for file_config in split_file_configs:
                
                cid = file_config['CID']
                if cid not in cid2index:
                    url_files.append(self.algocean.create_files({'hash': cid, 'type': 'ipfs'})[0])
                    cid2index[cid] = file_index
                    file_index += 1

                split_url_files_info[split].append(dict(
                    name=file_config['name'].split('/')[1],
                    type=file_config['type'],
                    size = file_config['size'],
                    file_index=cid2index[cid],
                    file_hash =self.algocean.web3.toHex((self.algocean.web3.keccak(text=cid)))
                ))


        self.url_files = url_files
        self.split_url_files_info = split_url_files_info
        # url_files = 

        return url_files, split_url_files_info
    @property
    def additional_information(self):

        info_dict = {
            'organization': 'huggingface',
            'package': {
                        'name': 'datasets',
                        'version': datasets.__version__
                        },
            'info': self.info, 
            'file_info': self.split_url_files_info
        }
        return info_dict

    def dispense_tokens(self,token=None):
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
            files, split_files_metadata = self.get_url_data()
        name = '.'.join([name, service_type])
        return self.algocean.create_service(name=name,
                                            timeout=timeout,
                                            datatoken=datatoken,
                                            datanft = datanft,
                                            service_type=service_type, 
                                            description= self.info['description'],
                                            files=files,
                                             additional_information=self.additional_information) 



    @property
    def metadata(self):
        st.sidebar.write(self.info)
        metadata ={}
        metadata['name'] = self.path
        metadata['description'] = self.info['description']
        metadata['author'] = self.algocean.wallet.address
        metadata['license'] = self.info.get('license', "CC0: PublicDomain")
        metadata['categories'] = []
        metadata['tags'] = []
        metadata['additionalInformation'] = self.info
        metadata['type'] = 'dataset'

        current_datetime = datetime.datetime.now().isoformat()
        metadata["created"]=  current_datetime
        metadata["updated"] = current_datetime

        return metadata
    @property
    def split_info(self):
        return self.info['splits']


    @property
    def datatokens(self):
        return self.get_datatokens(asset=self.asset)


    @property
    def wallet(self):
        return self.algocean.wallet

    @property
    def services(self):
        return self.asset.services


    @property
    def asset(self):
        st.write('BRO')
        assets =  self.algocean.search(text=f'metadata.name:{self.dataset_name} metadata.author:{self.wallet.address}')
        return assets[0]

    @property
    def datanft(self):
        if not hasattr(self,'_datanft'):
            self._datanft =  self.algocean.get_datanft(self.asset.nft)
        return self._datanft 

    @datanft.setter
    def datanft(self, value):
        self._datanft = value
        



    @staticmethod
    def get_cid_hash(file):
        cid =  os.path.basename(file).split('.')[0]
        cid_hash = module.algocean.web3.toHex((module.algocean.web3.keccak(text=cid)))
        return cid_hash

    def download(self, service=None, destination='bruh/'):
        if service == None:
            service = self.services[0]

        module.algocean.download_asset(asset=self.asset, service=service,destination=destination )
        

        
        for split,split_files_info in service.additional_information['file_info'].items():
            for file_info in split_files_info:
                file_index = file_info['file_index']
                file_name = file_info['name']
                did = self.asset.did

        # /Users/salvatore/Documents/commune/algocean/backend/bruh/datafile.did:op:6871a1482db7ded64e4c91c8dba2e075384a455db169bf72f796f16dc9c2b780,0
        # st.write(destination)
        # og_path = self.client.local.ls(os.path.join(destination, 'datafile.'+did+f',{file_index}'))
        # new_path = os.path.join(destination, split, file_name )
        # self.client.local.makedirs(os.path.dirname(new_path), exist_ok=True)
        # self.client.local.cp(og_path, new_path)

        files = module.client.local.ls(os.path.join(destination, 'datafile.'+module.asset.did+ f',{0}'))
                
            
    def create_asset(self, price_mode='free', services=None):

        self.datanft = self.algocean.create_datanft(name=self.dataset_name)

        metadata = self.metadata
        if services==None:
            services = self.create_service(price_mode=price_mode)



        if price_mode == 'free':
            self.dispense_tokens()

        if not isinstance(services, list):
            services = [services]
            


        return self.algocean.create_asset(datanft=self.datanft, metadata=metadata, services=services)






if __name__ == '__main__':
    import streamlit as st
    from algocean.utils import *

    # st.write(Dataset.__init__)
    module = DatasetModule()

    # module.algocean.create_datanft(name=module.builder_name)

    # st.write(isoformat2datetime('2022-08-24T20:53:57.342766')-datetime.timedelta(hours=1)) 
    # st.write(module.algocean.datatokens)
    st.write(module.algocean.services)

    # st.write(module.create_asset())
    # import time
    # trial_count = 0

    # except:
    #     print(f'Still waiting for {module.dataset_name}')
    # module.dispense_tokens()
    # st.write(module.algocean.get_balance(datatoken='token', datanft=module.dataset_name))
    # st.write(module.asset.services[0].additional_information['file_info'])
    # st.write(module.get_url_data())
    # module.download( )
    # module.algocean.save()
    # st.write(module.algocean.save())
    # # module.dataset.info.FSGSFS = 'bro'
    # load_dataset('glue')
    # st.write(module.info)
    # # st.write(module.save())
    # ds_builder = load_dataset_builder('ai2_arc')
    # with ray.init(address="auto",namespace="commune"):
    #     model = DatasetModule.deploy(actor={'refresh': False})
    #     sentences = ['ray.get(model.encode.remote(sentences))', 'ray.get(model.encoder.remote(sentences)) # whadup fam']
    #     print(ray.get(model.self_similarity.remote(sentences)))


# section 4 referencing section 4.18
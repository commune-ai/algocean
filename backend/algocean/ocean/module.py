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
from typing import *
# Create Alice's wallet

from ocean_lib.config import Config
from ocean_lib.models.data_nft import DataNFT
from ocean_lib.web3_internal.wallet import Wallet
from ocean_lib.web3_internal.constants import ZERO_ADDRESS
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
        self.datanfts = {}
        self.datatokens = {}
        self.dataassets = {}


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
        self.config['ocean'] = self.get_ocean(f'./config/{network}.in', return_ocean=False)
        self.ocean = Ocean(self.config['ocean'])
        self.web3 = self.ocean.web3

        

    
    def set_ocean(self, ocean_config):
        self.config['ocean'] = self.get_ocean(ocean_config, return_ocean=False)
        self.ocean = Ocean(self.config['ocean'])
        self.web3 = self.ocean.web3
        


    
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
        datanft = self.datanfts.get(nft_key)
        st.write(datanft)
        if datanft == None:
            datanft = self.ocean.create_data_nft(name=name, symbol=symbol, from_wallet=wallet)
            self.datanfts[nft_key] = datanft

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


        datanft_symbol= datanft.symbol()

        symbol = name if symbol is None else symbol
        key = '.'.join([datanft.symbol(), symbol])

        datatoken = self.datatokens.get(key)

        if datatoken == None:
            datatoken = datanft.create_datatoken(name=name, symbol=symbol, from_wallet=wallet)
            
            self.datatokens[key] = datatoken
        
        return self.datatokens[key]


    def get_contract(self, address:str, contract_class=ContractBase):
        return contract_class(web3=self.web3, address=address)
    
    def get_address(self, contract):
        return contract.address


    def load(self):
        self.load_state()
        # some loading post processing
        for k,v in self.datanfts.items():
            self.datanfts[k] = self.get_contract(address=v, contract_class=DataNFT)
        for k,v in self.datatokens.items():
            self.datatokens[k] = self.get_contract(address=v, contract_class=Datatoken)
        for k,v in self.dataassets.items():
            self.dataassets[k] = Asset.from_dict(v)


    def load_state(self):
        for k, v in self.config['load'].items():
            load_fn = getattr(getattr(self.client, v['module']), v['fn'])
            data = load_fn(**v['params'])
            if data == None:
                data = v.get('default', data)
            self.__dict__[k] = data

    # @staticmethod
    # def get_asset_did(asset:Asset):
    #     return asset.did

    def get_asset(self, asset) -> Asset:
        if isinstance(asset, Asset):
            return asset
        elif isinstance(asset, str):
            if asset.startswith('did:op:') :
                # is it a did
                return Asset(did=asset)
            else:
                # is it a key in sel.f
                assert asset in self.dataassets, f'Broooo: if you want to get the asset, your options are {list(self.dataassets.keys())}'
                return  self.dataassets[asset]

        


    def save(self):
        # some loading post processing
        for k,v in self.datanfts.items():
            self.datanfts[k] = self.get_address(contract=v)
        for k,v in self.datatokens.items():
            self.datatokens[k] = self.get_address(contract=v)
        for k,v in self.dataassets.items():
            self.dataassets[k] = v.as_dictionary()


        self.save_state()
        

    def save_state(self):
        for k, v in self.config['save'].items():

            data = self.__dict__[k]
            save_fn = getattr(getattr(self.client, v['module']), v['fn'])
            save_fn(**v['params'], data=data)

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
                        wallet=None):

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

    def get_datanft(self, datanft:Union[str, DataNFT]=None):
        '''
        dataNFT can be address, key in self.datanfts or a DataNFT
        '''
        if datanft == None:
            assert len(self.datanfts)>0
            return next(iter(self.datanfts.values()))

        if isinstance(datanft, str):
            if datanft in self.datanfts :
                datanft = self.datanfts[datanft]
            else:
                datanft = DataNFT(address=datanft)
        
        assert isinstance(datanft, DataNFT), f'datanft should be in the formate of DataNFT, not {datanft}'
        
        return datanft


    def add_service(self, datanft, datatoken, files, **kwargs):
        datanft = self.get_datanft(datanft)
        datatoken = self.get_datatoken(datanft)
        self.get_files()


    def create_asset(self,datanft, datatoken, files:list, wallet=None, **kwargs ):

        datanft = self.get_datanft(datanft=datanft)
        wallet = self.get_wallet(wallet)
        metadata = self.create_metadata(datanft=datanft, **kwargs.get('metadata', {}),wallet=wallet)


        if datanft in self.dataassets:
            return self.dataassets[datanft]
        
        datatoken = self.get_datatoken(datanft=datanft, datatoken=datatoken)
        deployed_datatokens = [datatoken]

        st.write(files)
        st.write(deployed_datatokens)
        default_kwargs= dict(
        data_nft_address = datanft.address,
        deployed_datatokens = deployed_datatokens,
        publisher_wallet= wallet,
        metadata= metadata,
        files=files
        )

        kwargs = {**kwargs, **default_kwargs}
        
        asset = self.ocean.assets.create(**kwargs)
        self.dataassets[datanft.symbol()] = asset
        return asset

    def list_dataassets(self, return_did=False):

        if return_did:
            # return dids
            return {k:v.did for k,v in self.dataassets.items()}
        else:
            # return keys only
            return list(self.dataassets.keys())
    
    @staticmethod
    def create_files(file_objects:Union[list, dict]):
        if isinstance(file_objects, dict):
            file_objects = [file_objects]
        assert isinstance(file_objects, list) 
        assert isinstance(file_objects[0], dict)

        output_files = []
        for file_object in file_objects:
            output_files.append(FilesTypeFactory(file_object))

        return output_files



    def dispense(self, datanft=None, datatoken=None, wallet=None):
        raise NotImplementedError

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
        

        datanft = self.get_datatonft(datnft)
        
        # if data token
        if isinstance(datatoken, Datatoken):
            return datatoken

        # if address
        elif isinstance(datatoken, str):

            if self.web3.isAddress(datatoken):
                datatoken_address = datatoken
                return self.ocean.get_datatoken(datatoken=datatoken_address)
            
            if datatoken in datatokens_map:
                return self.datatokens[datatoken]
            
            datatokens_map = self.nft_token_map(datanft=datanft)

            if datatoken in datatokens_map:
                return nft_token_map

                raise Exception('The datatoken is a string but must be an address or ')
        elif datatoken == None:
            datatokens_map = self.nft_token_map(datanft=datanft)
            assert len(datatokens_map)>0, f'THERE ARE NO TOKENS FOR NFT: name: {datanft.name()}'
            return next(iter(datatokens_map.values()))
        else:
            raise Exception(f'BRO {self.datanfts, self.datatokens}, {datatoken, datanft}')

    
    def get_balance(self,wallet:Union[Wallet,str]=None, datanft:str=None, datatoken:str=None):
        
        wallet_address = self.get_wallet(wallet=wallet, return_address=True)
        datatoken = self.get_datatoken(datanft=datanft, datatoken=datatoken )
        if datatoken == None:
            value =  self.web3.eth.get_balance(wallet_address)
        else:
            value =  datatoken.balanceOf(wallet_address)
        
        return value
        
    def list_services(self, asset):
        asset = self.get_asset(asset)
        return asset.services

    def get_service(self, asset=None, service=None):
        asset = self.get_asset(asset)
        if service == None:
            assert len(asset.services)>0, 'There are no services for the asset'
            return asset.services[0]
        elif isinstance(service, int):
            return asset.services[service]
        elif isinstance(service, str):
            raise NotImplementedError
        else:
            raise NotImplementedError
        
    def pay_for_access_service(self,
                              asset:Union[str,Asset],
                              service:Union[str,Service]=None,
                              wallet:Union[str, Wallet]=None, **kwargs):
        
        
        asset = self.get_asset(asset=asset)
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
        
    def download_asset(self, wallet, asset, service=None, destination='./', order_tx_id=None ):
        asset = self.get_asset(asset=asset)
        service= self.get_service(asset=asset, service=service)
        wallet = self.get_wallet(wallet=wallet) 

        if order_tx_id == None:
            order_tx_id = self.pay_for_access_service(asset=asset, service=service, wallet=wallet)

        file_path = self.ocean.assets.download_asset(
                                        asset=asset,
                                        service=service,
                                        consumer_wallet=wallet,
                                        destination=destination,
                                        order_tx_id=order_tx_id
                                    )
        return file_path

    def create_metadata(self, datanft=None, wallet=None, **kwargs ):

        wallet = self.get_wallet(wallet)

        metadata = {}

        metadata['description'] = kwargs.get('description', 'Insert Description')
        metadata['author'] = kwargs.get('author', wallet.address)
        metadata['license'] = kwargs.get('license', "CC0: PublicDomain")
        metadata['categories'] = kwargs.get('categories', [])
        metadata['tags'] = kwargs.get('tags', [])
        metadata['additionalInformation'] = kwargs.get('additionalInformation', {})
        metadata['type'] = kwargs.get('type', 'dataset')

        # the name must be a data nft or a custom name
        datanft = self.get_datanft(datanft)
        metadata['name'] = datanft.name

        created_datetime = datetime.datetime.now().isoformat()

        metadata["created"]=  created_datetime
        metadata["updated"] = created_datetime

        return metadata

    
        
    @classmethod
    def st_test(cls):
        module = cls()

        module.load()

        nft = 'NFT_IPFS'
        token = 'DT3'
        module.create_datanft(name=nft)
        datanft = module.get_datanft(nft)
        # st.write(nft.events.__dict__)
        module.create_datatoken(name=token, datanft=datanft)
        st.sidebar.write(datanft.contract.functions._functions)

        def get_tokens(self, datanft, return_type:str='value'):
            token_address_list = datanft.contract.caller.getTokensList()
            token_list = list(map(lambda t_addr: Datatoken(web3=self.web3,address=t_addr).contract, token_address_list))
            

            supporrted_return_types = ['map', 'key', 'value']

            assert return_type in supporrted_return_types, f'{return_type} not in {supporrted_return_types}'
            if return_type in ['map']:
                return {t.caller.name():t for t in token_list }
            elif return_type in ['key']:
                return [t.caller.name() for t in token_list]
            elif return_type in ['value']:
                return token_list
            
            else:
                raise NotImplementedError


        # module.create_dispenser(datatoken=token, datanft=nft)
        # module.dispense_tokens(datatoken=token, datanft=nft, wallet='bob', amount=50)
        
        
        # st.write(module.get_balance(datatoken=token, datanft=nft,  wallet=module.wallets['bob'].address))
        # # Specify metadata and services, using the Branin test dataset
 
        module.save()

        # if 'URL' in nft:
        #     url_files = module.create_files(dict(url="https://raw.githubusercontent.com/trentmc/branin/main/branin.arff", type='url'))
        # elif 'IPFS' in nft:
        #     cid = module.client.ipfs.put_json(data={'bro':1}, path='/tmp/fam.json')
        #     url_files = module.create_files([{'hash':f'{cid}', 'type':'ipfs'}]*1)

        # asset = module.create_asset(
        #     files=url_files,
        #     datanft=nft,
        #     datatoken=token
        # )




        # module.mint(
        #     datatoken=f'{nft}.{token}',
        #     to='bob', # can pass bobs wallet or address
        #     value=1.0
        # )

        # file_path = module.download_asset(
        #     asset=nft,
        #     wallet='bob',
        #     destination='./test_data',
        # )





if __name__ == '__main__':
    import os
    # OceanModule.st_test()

    module = OceanModule()
    module.st_test()

# Create Ocean instance
import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))

from algocean.utils import RecursiveNamespace, dict_put, dict_has

from ocean_lib.assets.asset import Asset
from ocean_lib.example_config import ExampleConfig
from ocean_lib.web3_internal.contract_base import ContractBase
from ocean_lib.models.datatoken import Datatoken
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.structures.file_objects import IpfsFile, UrlFile
from ocean_lib.services.service import Service
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
        self.data_nfts = {}
        self.data_tokens = {}
        self.data_assets = {}


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

    def create_data_nft(self, name:str , symbol:str, wallet:Union[str, Wallet]=None):
        wallet = self.get_wallet(wallet=wallet)

        nft_key =  symbol
        data_nft = self.data_nfts.get(nft_key)
        if data_nft == None:
            data_nft = self.ocean.create_data_nft(name=name, symbol=symbol, from_wallet=wallet)
            self.data_nfts[nft_key] = data_nft

        return data_nft

    def ensure_data_nft(self, data_nft:Union[str, DataNFT]): 
        if isinstance(data_nft, str):
            return self.data_nfts[data_nft]
        elif isinstance(data_nft, DataNFT):
            return data_nft
        else:
            raise Exception(f'The Data nft {data_nft} is not supported fam')

    def list_data_nfts(self):
        return list(self.data_nfts.keys())

    def create_datatoken(self, name:str, symbol:str, data_nft:Union[str, DataNFT]=None, wallet:Union[str, Wallet]=None):
        wallet = self.get_wallet(wallet)
        data_nft = self.ensure_data_nft(data_nft)

        nft_symbol = data_nft.symbol()
        key = '.'.join([nft_symbol, symbol])

        data_token = self.data_tokens.get(key)

        if data_token == None:
            datatoken = data_nft.create_datatoken(name=name, symbol=symbol, from_wallet=wallet)
            
            self.data_tokens[key] = datatoken
        
        return self.data_tokens[key]


    def get_contract(self, address:str, contract_class=ContractBase):
        return contract_class(web3=self.web3, address=address)
    
    def get_address(self, contract):
        return contract.address


    def load(self):
        self.load_state()
        # some loading post processing
        for k,v in self.data_nfts.items():
            self.data_nfts[k] = self.get_contract(address=v, contract_class=DataNFT)
        for k,v in self.data_tokens.items():
            self.data_tokens[k] = self.get_contract(address=v, contract_class=Datatoken)
        for k,v in self.data_assets.items():
            self.data_assets[k] = Asset.from_dict(v)


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
                assert asset in self.data_assets, f'Broooo: if you want to get the asset, your options are {list(self.data_assets.keys())}'
                return  self.data_assets[asset]

        


    def save(self):
        # some loading post processing
        for k,v in self.data_nfts.items():
            self.data_nfts[k] = self.get_address(contract=v)
        for k,v in self.data_tokens.items():
            self.data_tokens[k] = self.get_address(contract=v)
        for k,v in self.data_assets.items():
            self.data_assets[k] = v.as_dictionary()


        self.save_state()
        

    def save_state(self):
        for k, v in self.config['save'].items():

            data = self.__dict__[k]
            save_fn = getattr(getattr(self.client, v['module']), v['fn'])
            save_fn(**v['params'], data=data)

    @staticmethod
    def fill_default_kwargs(default_kwargs, kwargs):
        for k,v in default_kwargs.items():
            kwargs[k] =  kwargs.get(k, default_kwargs[k])
        return kwargs


    def data_nft_tokens(self, data_nft=None, return_keys=False):
        
        output_data_tokens = {}

        
        if data_nft == None:
            output_data_tokens =  self.data_tokens
        else:

            if isinstance(data_nft, DataNFT):
                target_key = data_nft.symbol()
            else:
                target_key = data_nft
            for k,v in self.data_tokens.items():
                k_nft,k_token = k.split('.')
                if target_key == k_nft:
                    output_data_tokens[k_token] = v

        if return_keys:
            return list(output_data_tokens.keys())
        return output_data_tokens


    def get_datanft(self, data_nft):
        '''
        dataNFT can be address, key in self.data_nfts or a DataNFT
        '''
        if isinstance(data_nft, str):
            if data_nft in self.data_nfts :
                data_nft = self.data_nfts[data_nft]
            else:
                data_nft = DataNFT(address=data_nft)
        
        assert isinstance(data_nft, DataNFT), f'data_nft should be in the formate of DataNFT, not {data_nft}'
        
        return data_nft


    def add_service(self, data_nft, data_token, files, **kwargs):
        data_nft = self.get_datanft(data_nft)
        data_token = self.get_datatoken(data_nft)
        self.get_files()


    def create_asset(self,data_nft, data_token, files:list, wallet=None, **kwargs ):

        data_nft = self.get_datanft(data_nft=data_nft)
        data_nft_symbol = data_nft.symbol()
        data_nft_address = data_nft.address
        wallet = self.get_wallet(wallet)
        metadata = self.create_metadata(data_nft=data_nft, wallet=wallet)


        if data_nft_symbol in self.data_assets:
            return self.data_assets[data_nft_symbol]
        
        data_token = self.get_datatoken(data_nft=data_nft_symbol, data_token=data_token)
        deployed_datatokens = [data_token]

        default_kwargs= dict(
        data_nft_address = data_nft_address,
        deployed_datatokens = deployed_datatokens,
        publisher_wallet= wallet,
        )

        kwargs = self.fill_default_kwargs(kwargs=kwargs, default_kwargs=default_kwargs)
        asset = self.ocean.assets.create(**kwargs)
        self.data_assets[data_nft_symbol] = asset
        return asset

    def list_data_assets(self, return_did=False):

        if return_did:
            # return dids
            return {k:v.did for k,v in self.data_assets.items()}
        else:
            # return keys only
            return list(self.data_assets.keys())
    
    @staticmethod
    def create_files(file_objects:Union[list, dict]):
        if isinstance(file_objects, dict):
            file_objects = [file_objects]
        assert isinstance(file_objects, list) 
        assert isinstance(file_objects[0], dict)

        output_files = []
        for file_object in file_objects:
            output_files.append(FilesTypeFactory.validate_and_create(file_object))

        return output_files



    def dispense(self, data_nft=None, data_token=None, wallet=None):
        raise NotImplementedError

    def mint(self, account:Union[str,Wallet], value:int=1,data_nft:str=None, data_token:str=None, token_address:str=None, wallet:Wallet=None , encode_value=True):
        wallet = self.get_wallet(wallet=wallet)
        account_address = self.resolve_account(account=account, return_address=True)

        if encode_value:
            value = self.ocean.to_wei(str(value))

        datatoken = self.get_datatoken(data_nft=data_nft,data_token=data_token, address=token_address)
        
        assert datatoken != None, f'data_token is None my guy, args: {dict(data_nft=data_nft, data_token=data_token, token_address=token_address)}'
        datatoken.mint(account_address=account_address, 
                        value=value, from_wallet=wallet )


    def get_datatoken(self, address:str=None, data_nft:str=None, data_token:str=None) -> Datatoken:
        

        if isinstance(data_token, Datatoken):
            return data_token

        if data_token in self.data_tokens:
            return self.data_tokens[data_token]

        if address != None:
            return self.ocean.get_datatoken(address)


        if data_nft != None or data_token != None:
            data_tokens_map = self.data_nft_tokens(data_nft=data_nft)
            assert data_token in data_tokens_map, f'{data_token} not in {list(data_tokens_map.keys())}'
            return data_tokens_map[data_token]
        else:
            return None
        

        assert False, f'BRO {self.data_nfts, self.data_tokens}, {data_token, data_nft}'

    def resolve_account(self, account:Union[Wallet, str], return_address:bool=False):
        '''
        resolves the account to default wallet if account is None
        '''

    
        account = self.wallet if account == None else account
        if account in self.wallets:
            account = self.wallet[account]

        if return_address:
            if isinstance(account, Wallet):
                account = account.address
            assert isinstance(account, str)
        else:
            assert isinstance(account, Wallet)
        
        return account

    
    def get_balance(self,account:Union[Wallet,str]=None, data_nft:str=None, data_token:str=None, token_address:str=None):
        
        account_address = self.resolve_account(account=account, return_address=True)
        data_token = self.get_datatoken(data_nft=data_nft, data_token=data_token, address=token_address)
        if data_token == None:
            value =  self.web3.eth.get_balance(account_address)
        else:
            value =  data_token.balanceOf(account_address)
        
        return value
        
    def list_services(self, asset):
        asset = self.get_asset(asset)
        return asset.services

    def get_service(self, asset, service=None):
        if isinstance(service, Service):
            return service
        asset = self.get_asset(asset)
        if service == None:
            return asset.services[0]
        elif isinstance(service, int):
            return asset.services[service]
        else:
            raise NotImplementedError
        
    def pay_for_access_service(self,
                              asset,
                              service=None,
                              consume_market_order_fee_address=None,
                              consume_market_order_fee_token=None,
                              consume_market_order_fee_amount=0,
                              wallet=None, **kwargs):
        asset = self.get_asset(asset=asset)
        service= self.get_service(asset=asset, service=service)
        wallet = self.get_wallet(wallet=wallet) 

        if consume_market_order_fee_token is None:
            consume_market_order_fee_token = service.datatoken
        if consume_market_order_fee_address is None:
            consume_market_order_fee_address = wallet.address
        
        default_kargs = dict(
            asset=asset,
            service=service,
            consume_market_order_fee_address=consume_market_order_fee_address,
            consume_market_order_fee_token=consume_market_order_fee_token,
            consume_market_order_fee_amount=consume_market_order_fee_amount,
            wallet=wallet,
        )



        order_tx_id = self.ocean.assets.pay_for_access_service(
            **default_kargs, **kwargs
        )     

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

    @staticmethod
    def create_metadata(self, data_nft=None, wallet=None, **kwargs ):

        wallet = self.get_wlalet(wallet)

        metadata = {}
        metadata['description'] = kwargs.get('description', 'Insert Description')
        metadata['author'] = kwargs.get('author', wallet.address)
        metadata['liscense'] = kwargs.get('liscense', "CC0: PublicDomain")
        metadata['categories'] = kwargs.get('categories', [])
        metadata['tags'] = kwargs.get('tags', [])
        metadata['additionalInformation'] = kwargs.get('additionalInformation', {})

        # the name must be a data nft or a custom name
        data_nft = self.get_datanft(data_nft)
        metadata['name'] = data_nft.name()

        created_datetime = datetime.datetime.now().isoformat()

        metadata["created"]=  created_datetime,
        metadata["updated"] = created_datetime

        return metadata

        
    @classmethod
    def st_test(cls):
        module = cls()

        module.load()
        nft_symbol = 'NFT_IPFS'
        token_symbol = 'DT3'

        module.create_data_nft(name='DataNFT1', symbol=nft_symbol)
        module.create_datatoken(name='DataToken1', symbol=token_symbol, data_nft=nft_symbol)

        # Specify metadata and services, using the Branin test dataset
 

        

        if 'URL' in nft_symbol:
            url_file = module.create_file(dict(url="https://raw.githubusercontent.com/trentmc/branin/main/branin.arff", type='url'))
        elif 'IPFS' in nft_symbol:
            cid = module.client.ipfs.put_json(data={'bro':1}, path='/tmp/fam.json')
            url_file = module.create_file({'hash':f'{cid}', 'type':'ipfs'})

        st.write(url_file.__dict__)
        asset = module.create_asset(
            metadata=metadata,
            files=[{'hash':f'{cid}', 'type':'ipfs'}],
            data_nft=nft_symbol,
            data_token=token_symbol
        )

        # module.save()
        # module.load()


        # Initialize Bob's wallet
        bob_wallet = module.wallets['bob']
        print(f"bob_wallet.address = '{bob_wallet.address}'")

        # Alice mints a datatoken into Bob's wallet
        module.mint(
            data_token=f'{nft_symbol}.{token_symbol}',
            account=bob_wallet.address, # can pass bobs wallet or address
            value=50
        )

        # Verify that Bob has ganache ETH
        module.get_balance(account=bob_wallet.address) > 0, "need ganache ETH"
        # two options of paying
        get_asset_option = 2

        # # Bob downloads. If the connection breaks, Bob can request again by showing order_tx_id.
        if get_asset_option == 1:
            file_path = module.download_asset(
                asset=nft_symbol,
                wallet=bob_wallet,
                destination='./')
        elif get_asset_option == 2:

            order_tx_id = module.pay_for_access_service(
                asset=nft_symbol,
                wallet=bob_wallet,
            )

            file_path = module.download_asset(
                asset=nft_symbol,
                wallet=bob_wallet,
                destination='./test_data',
                order_tx_id=order_tx_id
            )





if __name__ == '__main__':
    import os
    # OceanModule.st_test()

    module = OceanModule()
    module.st_test()
    # st.write(module.wallets)
    # st.write(module.network)



import os
import sys
from copy import deepcopy
sys.path.append(os.environ['PWD'])
from algocean.utils import dict_put, get_object, dict_has
from algocean import BaseModule
from ocean_lib.ocean.util import get_address_of_type, get_ocean_token_address, get_matic_token_address, get_web3



class ContractModule(BaseModule):

    default_config_path = 'contract.base'

    def __init__(self, config=None,  web3=None, **kwargs):

        BaseModule.__init__(self, config=config, **kwargs)
        self.web3 = web3


        
    @property
    def address(self):
        return self.contract.address


    @property
    def function_abi_map(self):
        return {f_abi['name']:f_abi for f_abi in self.abi}
    @property
    def function_names(self):
        return list(self.function_abi_map.keys())


    def call(self, function, args=[]):
        if len(args) == 0:
            args.append({'from': self.account})
        output = getattr(self.contract, function)(*args)
        return self.parseOutput(function=function, outputs=output)


    def parseOutput(self, function, outputs):
        output_abi_list = self.function_abi_map[function]['outputs']
        
        parsedOutputs = {}
        for i,output_abi  in enumerate(output_abi_list) :
            output_key = i 
            if output_abi['name']:
                output_key = output_abi['name']
            
            parsedOutputs[output_key] = outputs[i]
            if 'components' in output_abi:
                component_names = [c['name'] for c in output_abi['components']]
                
                parseStruct = lambda o:  dict(zip(component_names, deepcopy(o)))
                if type(outputs[i]) in [list, tuple, set]:
                    parsedOutputs[output_key] = list(map(parseStruct, outputs[i]))
                else:
                    parsedOutputs[output_key] = parseStruct(outputs[i])
        
        return parsedOutputs

    artifacts_path = f'{os.environ["PWD"]}/artifacts/'

    contracts_path = f'{os.environ["PWD"]}/contracts/'
    @property
    def contract_paths(self):
        return list(filter(lambda f: f.endswith('.sol'), self.client.local.glob(self.contracts_path+'**')))

    def get_artifact(self, path):
        available_abis = self.contracts + self.interfaces

        if path in self.contracts:
            root_dir = os.path.join(self.artifacts_path, 'contracts')
        elif path in self.interfaces:
            root_dir = os.path.join(self.artifacts_path, 'interfaces')
        else:
            raise Exception(f"{path} not in {available_abis}")
        json_name = os.path.basename(path).replace('.sol', '.json')

        artifact_path = os.path.join(root_dir, path, json_name)
        artifact = self.client.local.get_json(artifact_path)
        return artifact


    def get_abi(self,path):
        return self.get_artifact(path)['abi']
    interfaces_path = f'{os.environ["PWD"]}/interfaces/'
    @property
    def interface_paths(self):
        return list(filter(lambda f: f.endswith('.sol'),self.client.local.glob(self.interfaces_path+'**')))


    @property
    def artifact_paths(self): 
        full_path_list = list(filter(lambda f:f.endswith('.json') and not f.endswith('dbg.json') and os.path.dirname(f).endswith('.sol'),
                            self.client.local.glob(f'{self.artifacts_path}**')))
        
        
        return full_path_list
    
    @property
    def artifacts(self):
        artifacts = []
        for path in self.artifact_paths:
            simple_path = deepcopy(path)
            simple_path = simple_path.replace(self.artifacts_path, '')
            artifacts.append(simple_path)
        return artifacts

    def set_network(self, url='LOCAL_NETWORK_RPC_URL'):
        url = os.getenv(url, url)
        self.web3 = Web3(Web3.HTTPProvider(url))
        return self.web3
    connect_network = set_network

    def set_account(self, private_key):
        private_key = os.getenv(private_key, private_key)
        self.account = AccountModule(private_key=private_key)
        return self.account
    
    def compile(self):
        # compile smart contracts in compile
        return self.run_command('npx hardhat compile')
        
    @property
    def network_modes(self):
        return list(self.network_config.keys())

    @property
    def available_networks(self):
        return ['local', 'ethereum']

    @property
    def network_config(self):
        network_config_path = f'{self.root}/web3/data/network-config.yaml'
        return self.client.local.get_yaml(network_config_path)

    @property
    def contracts(self):
        contracts = list(filter(lambda f: f.startswith('contracts'), self.artifacts))
        return list(map(lambda f:os.path.dirname(f.replace('contracts/', '')), contracts))
    @property
    def interfaces(self):
        interfaces = list(filter(lambda f: f.startswith('interfaces'), self.artifacts))
        return list(map(lambda f:os.path.dirname(f.replace('interfaces/', '')), interfaces))
if __name__ == '__main__':
    import streamlit as st
    module = ContractBaseModule.deploy(actor=False)
    
    with st.sidebar.expander('Deploy', True):
        contract_options =  module.contracts
        path2idx_contract =  {p:i for i,p in enumerate( module.contracts)}
        st.selectbox('Select a Contract',  contract_options, path2idx_contract['token/ERC20/ERC20.sol'])
        
        

        
        st.selectbox('Select a Network', module.available_networks, 0)
    
    
    with st.expander('Select Network', True):
        network_mode_options = module.network_modes
        selected_network_mode = st.selectbox('Select Mode', network_mode_options, 1)
        
        if selected_network_mode == 'live':
            network2endpoints = {config['name']:config['networks'] for config in module.network_config[selected_network_mode]}
            selected_network = st.selectbox('Select Network', list(network2endpoints.keys()), 4)

            endpoint2info = {i['name']:i for i in network2endpoints[selected_network]}  
            selected_endpoint = st.selectbox('Select Endpoint', list(endpoint2info.keys()) , 4)
            network_info = endpoint2info[selected_endpoint]
        elif selected_network_mode == 'development':
            network2info = {config['name']:config for config in module.network_config[selected_network_mode]}
            selected_network = st.selectbox('Select Network', list(network2info.keys()), 4)
            network_info = network2info[selected_network]
        else:
            raise NotImplemented

        # st.write(module.run_command("env").stdout)
        
        st.write(network_info)
        from algocean.account import AccountModule
        # os.environ["PRIVATE_KEY"] = "0x8467415bb2ba7c91084d932276214b11a3dd9bdb2930fefa194b666dd8020b99"
        account = AccountModule(private_key='PRIVATE_KEY')
        st.write(account.account)
        def build_command( network_info):
            cmd = network_info['cmd']
            cmd_settings = network_info['cmd_settings']

            for k,v in cmd_settings.items():
                cmd += f' --{k} {v}'
            
            return  cmd



        # st.write(module.run_command(build_command(network_info))) 
        # st.write(module.run_command('npx hardhat node', False))
        # st.write(network_info['cmd'])
        import subprocess
        from subprocess import DEVNULL, PIPE
        import psutil

        # BaseModule.kill_port(8545)

        def launch(cmd: str, **kwargs) -> None:
            print(f"\nLaunching '{cmd}'...")
            out = DEVNULL if sys.platform == "win32" else PIPE

            return psutil.Popen(['bash','-c', cmd], stdout=out, stderr=out)



        cmd = build_command(network_info)
        # # p = subprocess.Popen([sys.executable, '-c', f'"{cmd}"'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # p = launch(cmd)
        # import time
        # while True:
        #     time.sleep(2)
        #     st.write( p.stderr.read(1000))
        # st.write(module.run_command('npx hardhat node'))

        # p = subprocess.call(['ls'], stdout=subprocess.PIPE,
        #                                     stderr=subprocess.PIPE,
        #                                     shell=True)

        # st.write(module.run_command(cmd))

        st.write(os.getenv('GANACHE_URL'))
        # # import yaml
        # st.write(module.compile())
        # st.write(module.client.local.get_yaml(f'{module.root}/web3/data/network-config.yaml'))
        # st.write(module.get_abi('token/ERC20/ERC20.sol'))
        # st.write(module.get_abi('dex/sushiswap/ISushiswapFactory.sol'))




import os
import sys
sys.path.append(os.environ['PWD'])
from algocean.utils import dict_put, get_object, dict_has
from algocean import BaseModule
from copy import deepcopy


class ContractBaseModule(BaseModule):

    default_config_path = 'contract.base'

    def __init__(self, config=None,  web3=None, **kwargs):

        BaseModule.__init__(self, config=config, **kwargs)
        
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
    st.write(module.interfaces)
    st.write(module.get_abi('dex/sushiswap/ISushiswapFactory.sol'))
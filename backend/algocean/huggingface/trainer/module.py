
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
import json
import os
from algocean.utils import dict_put
from datasets.utils.py_utils import asdict, unique_values
import datetime
import pyarrow
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ocean_lib.models.data_nft import DataNFT
import fsspec
import os
# from ipfsspec.asyn import AsyncIPFSFileSystem
from fsspec import register_implementation
import asyncio
import io
from algocean.ocean import OceanModule
from algocean.utils import Timer, round_sig


from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, load_dataset_builder, load_metric




class TrainerModule(BaseModule):
    default_cfg_path = 'huggingface.trainer.module'

    datanft = None
    default_token_name='token' 
    last_saved = None

    
    dataset = {}
    def __init__(self, config:dict=None, override:dict={}):
        BaseModule.__init__(self, config=config)
        self.algocean = OceanModule()
        self.web3 = self.algocean.web3
        self.ocean = self.algocean.ocean
        self.override_config(override)
        self.hub = self.get_object('huggingface.hub.module.HubModule')()
        self.load_state()

    def load_state(self):
        if self.config.get('model_only'):
            self.load_model()
            pass
        else:
            self.load_model()
            self.load_dataset()
            self.load_optimizer()
            self.load_metric()
            self.load_schedular()
            
    def load_dataset(self):
        dataset_class = self.get_object('huggingface.dataset.module.DatasetModule')
        override = dict(dataset=self.config.get('dataset'), load_dataset=True, client=self.client)
        self.dataset = dataset_class(override=override).dataset
        self.load_pipeline()
        self.dataloaders = self.get_dataloaders()
        # st.write(self.dataset.dataset)

    def split(self, split):
        return self.dataset[split]


    def load_pipeline(self):
        kwargs = self.config.get('pipeline')
        pipeline_class = self.import_object(path=kwargs.get('module'))
        pipeline = getattr(pipeline_class, kwargs['init']['fn'])(**kwargs['init'].get('kwargs', {}))
        self.pipeline = pipeline
        
        def pipeline_function(examples):
            return pipeline(examples["text"], padding="max_length", truncation=True)
        
        for split, dataset in self.dataset.items():
            dataset = dataset.map(pipeline_function, batched=True)
            dataset = dataset.remove_columns(["text"])
            dataset = dataset.rename_column("label", "labels")    
            dataset.set_format('torch')
            self.dataset[split] = dataset
    
    @property
    def device(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return device

    def load_model(self):
        kwargs = self.config.get('model')
        model_class = self.import_object(path=kwargs.get('module'))
        model = getattr(model_class, kwargs['init']['fn'])(**kwargs['init'].get('kwargs', {}))
        self.model = model
        self.model.to(self.device)


    def load_metric(self):
        self.metric = load_metric(self.config['metric'])
        
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)




    @property
    def num_epochs(self):
        for k in ['num_epochs', 'epochs']:
            v = self.config.get(k)
            if isinstance(v, int):
                return v
        
        raise Exception('Please specify the number of epochs in the config via num_epochs, or epochs')

    @property
    def num_training_steps(self):
        return self.num_epochs * len(self.dataloader['train'])
        

    def load_optimizer(self):
        kwargs = self.config.get('optimizer')
        optimizer_class = self.import_object(kwargs['module'])
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs['params'])

    @property
    def batch_size(self):
        return self.config.get('batch_size', 8) 

    @property
    def shuffle(self):
        return self.config.get('shuffle', True)
    
    @property
    def wallet(self):
        account = self.algocean.wallet
        return account

    account = wallet
    def get_dataloaders(self):
        self.dataloader = {}
        for split in self.dataset.keys():
            self.dataloader[split] = DataLoader(self.dataset[split], shuffle=self.shuffle, batch_size=self.batch_size)
        return self.dataloader
    def load_schedular(self):
        kwargs = self.config['schedular']
        get_scheduler = self.import_object(kwargs['module'])
        self.scheduler = get_scheduler( optimizer=self.optimizer, num_training_steps=self.num_training_steps, **kwargs['params'])

    def train(self):
        progress_bar = tqdm(range(self.num_training_steps))
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, batch in enumerate(self.dataloader['train']):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                st.write(f"EPOCH: {epoch} Batch: {i} loss: {loss}")
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                # progress_bar.update(1)


    @staticmethod
    def obj2str(data):
        data_str = None
        if isinstance(data, dict):
            data_str =  json.dumps(data)
        if isinstance(data, list):
            data_str = json.dumps(data)
        elif type(data) in [bool, int]:
            data_str = str(data)
        elif type(data) in [str]:
            data_str = data
        else:
            raise NotImplementedError
        
        return data_str

    def hash(self, data, web3=None):
        if web3 == None:
            web3 = self.web3
        
        data_str = self.obj2str(data=data) 
        
        return web3.keccak(text=data_str)
        


    def sign_message(self, data, account=None, web3=None):
    
        from hexbytes.main import HexBytes
        from eth_account.messages import encode_defunct


        if web3 == None:
            web3 = self.web3

        if account == None:
            account = self.account


        
        msg = encode_defunct(text=data)
        msg = web3.eth.account.sign_message(msg, private_key=account.private_key )
        return_dict =  msg._asdict()
        for k,v in return_dict.items():
            if isinstance(v, HexBytes):
                return_dict[k] =  v.hex()

        return return_dict

    def get_sample(self, split='train', **kwargs):
        return next(iter(self.dataloader[split]))

    
    def get_onnx(self):
        dummy_model_input = self.get_sample()
        dummy_model_input.pop('labels')
        self.model = self.model.to('cpu')
        torch.onnx.export(
                        self.model, 
                        tuple(dummy_model_input.values()), 
                        f="torch-model.onnx",  
                        input_names=['input_ids', 'attention_mask'], 
                        output_names=['logits'], 
                        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                                    'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                                    'logits': {0: 'batch_size', 1: 'sequence'}}, 
                        do_constant_folding=True, 
                        opset_version=13, 
                    )

    ''' PyTorch backend model factory '''
    def serialize(self, model=None):
        ''' Serialize PyTorch model to JSON message '''
        # metadata = {}
        # metadata_file = os.path.join(os.path.dirname(__file__), 'onnx-metadata.json')
        # with open(metadata_file, 'r', encoding='utf-8') as file:
        #     for item in json.load(file):
        #         name = 'onnx::' + item['name']
        #         metadata[name] = item

        if model == None:
            model = self.model
        json_model = {}
        json_model['signature'] = 'netron:pytorch'
        json_model['format']  = 'TorchScript'
        json_model['graphs'] = []
        json_graph = {}
        json_graph['arguments'] = []
        json_graph['nodes'] = []
        json_graph['inputs'] = []
        json_graph['outputs'] = []
        json_model['graphs'].append(json_graph)
        data_type_map = dict([
            [ torch.float16, 'float16'], # pylint: disable=no-member
            [ torch.float32, 'float32'], # pylint: disable=no-member
            [ torch.float64, 'float64'], # pylint: disable=no-member
            [ torch.int32, 'int32'], # pylint: disable=no-member
            [ torch.int64, 'int64'], # pylint: disable=no-member
        ])
        arguments_map = {}
        def argument(value):
            if not value in arguments_map:
                json_argument = {}
                json_argument['name'] = str(value.unique()) + '>' + str(value.node().kind())
                if value.isCompleteTensor():
                    json_tensor_shape = {
                        'dimensions': value.type().sizes()
                    }
                    json_argument['type'] = {
                        'dataType': data_type_map[value.type().dtype()],
                        'shape': json_tensor_shape
                    }
                if value.node().kind() == "prim::Param":
                    json_argument['initializer'] = {}
                arguments = json_graph['arguments']
                arguments_map[value] = len(arguments)
                arguments.append(json_argument)
            return arguments_map[value]

        for input_value in model.inputs():
            json_graph['inputs'].append({
                'name': input_value.debugName(),
                'arguments': [ argument(input_value) ]
            })
        for output_value in model.outputs():
            json_graph['outputs'].append({
                'name': output_value.debugName(),
                'arguments': [ argument(output_value) ]
            })
        for node in model.nodes():
            kind = node.kind()
            json_type = {
                'name': kind
            }
            json_node = {
                'type': json_type,
                'inputs': [],
                'outputs': [],
                'attributes': []
            }
            json_graph['nodes'].append(json_node)
            for name in node.attributeNames():
                value = node[name]
                json_attribute = {
                    'name': name,
                    'value': value
                }
                if torch.is_tensor(value):
                    json_node['inputs'].append({
                        'name': name,
                        'arguments': []
                    })
                else:
                    json_node['attributes'].append(json_attribute)

            for input_value in node.inputs():
                json_parameter = {
                    'name': 'x',
                    'arguments': [ argument(input_value) ]
                }
                json_node['inputs'].append(json_parameter)

            for output_value in node.outputs():
                json_node['outputs'].append({
                    'name': 'x',
                    'arguments': [ argument(output_value) ]
                })

        text = json.dumps(json_model, ensure_ascii=False)
        return text.encode('utf-8')


if __name__ == '__main__':
    import streamlit as st
    import numpy as np
    from algocean.utils import *
    module = TrainerModule(override={'model_only':False})

    web3 = module.web3
    st.write(module.account)
    children = []
    for m in module.model.children():
        
        children += [m]
    
    # for m in children[0].children():
    #     st.write(m.inputs)
        
    # st.write(module.get_onnx())

    # import web3
    # data = {k:v.tolist() for k,v in module.model.state_dict().items()}
    # data = {k: data[k] for k in list(data.keys())[:10]}
    # ModelFactory().serialize(module.model)
    # st.write(module.model)
    # st.write(module.hash(data = data))
    # st.write( module.sign_message(data=data))
    # module.train()
    # st.write(module.model.state_dict().keys())
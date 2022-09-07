
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
    __file__ = __file__
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
        if self.config.get('load_state', False):
            pass
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
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].shard(20,1)

        self.dataloaders = self.get_dataloaders()
        # st.write(self.dataset.dataset)

    def split(self, split):
        return self.dataset[split]

    @property
    def splits(self):
        return list(self.dataset.keys())

    def load_pipeline(self):
        kwargs = self.config.get('pipeline')
        pipeline_class = self.import_object(path=kwargs.get('module'))
        pipeline = getattr(pipeline_class, kwargs['init']['fn'])(**kwargs['init'].get('kwargs', {}))
        self.pipeline = pipeline
        
        def pipeline_function(examples):
            return pipeline(examples['sentence'], **kwargs.get('params'))
        
        for split, dataset in self.dataset.items():
            dataset = dataset.map(pipeline_function, batched=True)
            dataset = dataset.remove_columns(kwargs['features']['remove'])
            for k, v in kwargs['features']['map'].items():
                dataset = dataset.rename_column(k, v) 

           
            dataset.set_format(kwargs['format'])
            self.dataset[split] = dataset
        self.meta_keys =  kwargs['features'].get('meta', [])
        self.input_keys =  kwargs['features'].get('input', [])
        if len(self.input_keys) == 0:
            non_sample_keys = kwargs['features']['remove'] + kwargs['features']['meta']
            self.input_keys = [k for k in self.sample_example if k not in non_sample_keys]


    def resolve_split(self, split=None):
        if split == None:
            split = self.splits[0]
        assert isinstance(split, str)
        return split
        
    @property
    def sample_example(self):
        return self.sample()

    def sample(self, split=None, idx=0):
        split = self.resolve_split(split)
        return self.dataset[split][idx]
    
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

    @property
    def batches_per_epoch(self):
        return self.config.get('batches_per_epoch')


    def step(self, split='train',nograd=False, proof=True, step=0):

        if split == 'train':
            batch = next(iter(self.dataloader['train']))
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # st.write(self.model.forward)
            st.write(get_function_schema(self.model))
            outputs = self.model(**{k:batch[k] for k in self.input_keys})
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            


        elif split != 'train' or nograd == True:
            with torch.nograd():
                batch = next(self.dataloader[split])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss


        if proof:
            output = outputs.__dict__
            metadata_batch = {k:batch[k] for k in self.meta_keys }
            output['meta'] = metadata_batch
            output['weights'] = self.model.state_dict()
            # st.write(metadata_batch)
            st.write(dir(outputs))
            outputs = {**outputs.__dict__, **metadata_batch}

        return outputs

    def train(self):
        progress_bar = tqdm(range(self.num_training_steps))
        self.model.train()
        for epoch in range(self.num_epochs):

            for i in range(self.batches_per_epoch):
                output_dict = self.step(split=True)
                st.write(f"EPOCH: {epoch} Batch: {i} loss: {output_dict['loss']}")

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
        st.write(len(self.dataloader[split]))
        return next(iter(self.dataloader[split]))

    @property
    def default_onnx_path(self):
        default_onnx_path = f"/tmp/onnx/{__file__}/torch-model.onnx"
        self.client.local.makedirs(os.path.dirname(default_onnx_path),True)
        return default_onnx_path
    def save_onnx(self, path=None):
        dummy_model_input = self.get_sample()
        dummy_model_input.pop('labels')
        self.model = self.model.to('cpu')
        if path == None:
            path = self.default_onnx_path
            
        torch.onnx.export(
                        self.model, 
                        tuple(dummy_model_input.values()), 
                        f=path,  
                        input_names=['input_ids', 'attention_mask'], 
                        output_names=['logits'], 
                        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                                    'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                                    'logits': {0: 'batch_size', 1: 'sequence'}}, 
                        do_constant_folding=True, 
                        opset_version=13, 
                    )

        self.saved_onnx_path = path


    def load_onnx(self, path=None):
        if path == None:
            path = self.default_onnx_path
        import onnx
        return onnx.load(path)

    ''' PyTorch backend model factory '''
    def serialize(self, model=None, path=None):
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


        # text = json.dumps(json_model, ensure_ascii=False)
        # return text.encode('utf-8')
        if isinstance(path, str):
            text = json.dumps(json_model, ensure_ascii=False)
            return text.encode('utf-8')
        return json_model
    def follow_grad(grad_fn):

        '''
        TODO: NOT READY
        '''
        while True:
            try:
                if len(grad_fn.next_functions)>1:
                    for next_grad_fn in grad_fn.next_functions:
                        follow_grad(next_grad_fn)
                    continue
                elif len(grad_fn.next_functions)==1:
                    tmp_grad_fn = grad_fn.next_functions[0][0]
                else:
                    raise IndexError
                st.write(tmp_grad_fn.name())
                grad_fn = tmp_grad_fn

            except IndexError as e :
                break

if __name__ == '__main__':
    import streamlit as st
    import numpy as np
    from algocean.utils import *

    module = TrainerModule(override={'model_only':False, 'load_state': True})
    proof = module.step()
    st.write(proof.keys())

    
    # # # st.write(module.save_onnx())
    # # onnx_model = module.load_onnx()
    # # st.write(module.model.bert)
    # st.write(module.hub.list_models())
    # web3 = module.web3
    # st.write(module.account)
    # children = []
    # for m in module.model.named_parameters():
        
    #     children += [m]
    #     st.write(m[0], m[1].shape)
    # json_model = module.get_onnx()
    
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

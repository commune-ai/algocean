import hub
import os, sys
sys.path.append(os.environ['PWD'])

from typing import List,Optional
from dataclasses import dataclass,field
import json
from pathlib import Path
import numpy as np
from algocean import BaseModule







@dataclass
class ActiveLoopModule(BaseModule):
    src:str
    api_key:str= field(init=False,repr=False,default="")
    path:Optional[str]=None
    train:Optional[str]=None
    test:Optional[str]=None

    def __init__(self, config=None, override={}):
        BaseModule.__init__(self, config=config)
        self.override_config(override)

    
    def load(self,splits=["train","test"],filter=False,token=None):


        for split in splits:
            if split in ['train', 'test']:
                split_ds = "-".join([self.path,split])
                url = "/".join([self.src,split_ds])
                setattr(self, split,hub.load(url,token=self.api_key))

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


    def _parse_metadata(self,metadata):

        metadata_dict = {}


        select_fields = ["chunk_compression","sample_compression","dtype","htype","length","max_chunk_size","max_shape","min_shape","name","class_names"]


        parse_topkey = {k: v for k, v in metadata.items() if k is not None and not k.startswith("_")}

        for k_top in parse_topkey:
            
            metadata_dict.update({k_top:{k: v for k, v in parse_topkey[k_top].items() if k in select_fields}})

        return metadata_dict


    def create_tensors(self,labels:list,htype:list,compression:list,class_names:list):

        for label,htype,compression,class_names in zip(labels,htype,compression,class_names):
            ds.create_tensor(label,htype=htype, sample_compression =compression)

    def populate_ds(self):
        with ds:
            # Iterate through the files and append to hub dataset
            for file in files_list:
                label_text = os.path.basename(os.path.dirname(file))
                label_num = class_names.index(label_text)

                #Append data to the tensors
                ds.append({'images': hub.read(file), 'labels': np.uint32(label_num)})

    @hub.compute
    def filter_labels(sample_in, labels_list):

        return sample_in.labels.data()['text'][0] in labels_list

    def max_array_length(arr_max, arr_to_compare):  # helper for __str__
        for i in range(len(arr_max)):
            str_length = len(arr_to_compare[i])
            if arr_max[i] < str_length:
                arr_max[i] = str_length
        return arr_max


    def summary_dataset(dataset):

        metadata = {}
        tensor_dict = dataset.tensors

        for tensor_name in tensor_dict:

            metadata[tensor_name] = {}

            tensor_object = tensor_dict[tensor_name]

            tensor_htype = tensor_object.htype
            if tensor_htype == None:
                tensor_htype = "None"

            shape = tensor_object.shape
            tensor_shape = str(tensor_object.shape_interval if None in shape else shape)

            tensor_compression = tensor_object.meta.sample_compression
            if tensor_compression == None:
                tensor_compression = "None"

            tensor_dtype = tensor_object.dtype
            if  tensor_dtype == None:
                tensor_dtype = "None"

            #Working - Improvement to resolve - ValueError: dictionary update sequence element #0 has length 10; 2 is required
            if tensor_name == "labels":

                label_distr = np.unique(tensor_object.data()["value"],return_counts=True)
                metadata[tensor_name].update({"label_distribution":label_distr})

            metadata[tensor_name].update({"htype":tensor_htype})
            metadata[tensor_name].update({"shape":tensor_shape})
            metadata[tensor_name].update({"compression":tensor_compression})
            metadata[tensor_name].update({"dtype":tensor_dtype})


        return metadata




if __name__ == '__main__':
    import streamlit as st
    # from metadata import DatasetMetdata
    import os

    ds_select = st.selectbox("Active Loop Datasets",["fashion-mnist","mnist",
                                        "coco","imagenet","cifar10",
                                        "cifar100"])

    AL = ActiveLoopModule("hub://activeloop",path=ds_select)
    AL.load_ds()

    st.write("Active Loop Train Dataset")
    st.write(AL)

    info = AL.train.info



    w = summary_dataset(AL.train)

    st.write(w)



    #WIP
    st.write("Manual Dataset, still work in progress")
    # AL_manual = ActiveLoopDS(src="app/animals",path="./animals_hub")
    # manual_ds = AL_manual.automatic_ds()
    #
    # st.write(AL_manual)
    #
    # st.write(summary_dataset(manual_ds))
    #
    # md_parsed = AL_manual.get_dataset_metadata("animals_hub",parse=True)
    # st.write("Parsed Manual Metadata")
    # st.wrtie(md_parsed)
    #
    # md = AL_manual.get_dataset_metadata("animals_hub",parse=False)
    # st.write("UnParsed Manual Metadata")
    # st.write(md)

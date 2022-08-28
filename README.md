<div align="center">

# **Algocean** <!-- omit in toc -->

Actually Democratizing Data and AI

</div>

***
## Summary 

Algocean is a combination of Ocean, Huggingface,  Algovera and Commune Toolings. This will serve as an sdk for developers to share and monetize from their datasets and models in a framework agnostic way. 
***

## Setup

1. Clone Repo and its Submodules

```
git clone https://github.com/commune-ai/algocean.git
cd algocean
git submodule update --init --recursive
```

2. Spinnup Docker Compose which starts barge and backend
```
make up
```

3. Run the Streamlit app with huggingface dataset
```
make streamlit arg=huggingface/dataset
```
after running this command, view the streamlit app in [localhost:8501](localhost:8501)





## Commands

- Run 
    
     ```make up```
-  Enter Backend 
    
     ``` make bash arg=ocean_backend_1```
-  Enter Anything Else 
    
     ``` make bash arg=INSERT_CONTAINER_NAME```


- Run Streamlit Server
    
     ``` make app arg=${path to streamlit in backend with respect to algocean}```

     [localhost:8501](localhost:8501)

**Base Modules**
- ClientModule
    - manages a connection to various clients that can help store and retreive data from different databases

    Supported Clients:
        
        - ipfsspec
        - local

- ConfigLoader
    - A module that is dedicated to loading config files with some additional funcitonality that allows for composing and referencing multiple configs and variables within configs


- BaseModule
    - The base process that connects to configloader, clients, with a base module



**Specifc Modules**

- OceanModule (WIP)
- HuggingFace Modules (WIP)
    - base
    - dataset
    - model
    - process
    - trainer

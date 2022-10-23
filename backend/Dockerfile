

# syntax=docker/dockerfile:1
FROM opentensorfdn/bittensor:3.3.3-torch1.12.0-cuda11.3-pm2
# Install cubit

# Install cubit
COPY ./cubit /cubit

RUN cd /cubit && python3 -m pip install -r requirements.dev.txt
RUN cd /cubit && python3 setup.py build_ext --inplace

ENV PYTHONPATH "${PYTHONPATH}:/cubit"

WORKDIR /app
ADD ./bittensor /app/bittensor
RUN python3 -m pip install -e /app/bittensor

RUN python3 -m pip install -U "ray[default]"
RUN python3 -m pip install -U streamlit
RUN python3 -m pip install -U gradio
RUN python3 -m pip install -U plotly
RUN python3 -m pip install -U sentence-transformers
RUN python3 -m pip install -U datasets
RUN python3 -m pip install hub

ADD ./ocean.py /app/ocean.py
RUN python3 -m pip install -e ./ocean.py

ADD ./ipfspy /app/ipfspy
RUN python3 -m pip install -e ./ipfspy

ADD ./ipfsspec /app/ipfsspec
RUN python3 -m pip install -e ./ipfsspec

# RUN python3 -m pip install torchsort

RUN python3 -m pip install hub



RUN apt-get install curl -y
# RUN curl -sSL https://get.docker.com/ | sh
RUN apt-get update && \
    apt-get -qy full-upgrade && \
    apt-get install -qy curl && \
    apt-get install -qy curl && \
    curl -sSL https://get.docker.com/ | sh
# RUN conda activate algocean
# add BITTENSOR commune fork


ENV NODE_VERSION=16.17.1
RUN apt install -y curl
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"
RUN node --version
RUN npm --version

# fix npm - not the latest version installed by apt-get

# RUN cp hardhat.config.barge.js hardhat.config.js
# ENV NETWORK=barge
# ENV NETWORK_RPC_URL=127.0.0.1:8545
# RUN npx hardhat compile
# ENTRYPOINT ["/ocean-contracts/scripts/deploy_docker.sh"

RUN npm install --save-dev hardhat
RUN npm install --save-dev @nomicfoundation/hardhat-toolbox

COPY hardhat.config.js .
RUN npx hardhat
RUN npm install @openzeppelin/contracts
RUN npm install @uniswap/v3-periphery
RUN npm install @uniswap/v2-periphery


COPY ./scripts .

RUN python3 -m pip install -U accelerate
RUN python3 -m pip install -U diffusers


ADD ./brownie /app/brownie
RUN python3 -m pip install -e ./brownie

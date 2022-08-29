
FROM python:3.8-slim-buster

# # Install linux dependencies
# RUN apt-get install -y npm
# # install ganache for simulating evm vm
# RUN npm install -g ganache-cli

WORKDIR /app

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

# add BITTENSOR commune fork
ADD ./ocean.py /app/ocean.py
RUN pip install -e ./ocean.py

# add ipfspy commune fork
ADD ./ipfspy /app/ipfspy
RUN pip install -e ./ipfspy

# add ipfsspec commune fork
ADD ./ipfsspec /app/ipfsspec
RUN pip install -e ./ipfsspec
RUN pip install jupyterlab
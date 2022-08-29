#!/usr/bin/env bash

eval docker build -t algocean/subgraph ./subgraph
eval docker build -t algocean/provider ./provider
eval docker build -t algocean/aquarius ./aquarius
eval docker build -t algocean/contracts ./contracts
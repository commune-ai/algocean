#/bin/bash
conda activate algocean
ray start --head;
apt-get update && apt-get install procps;
apt-get install lsof;


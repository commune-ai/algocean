#!/usr/bin/env bash
#
# Copyright (c) 2020 Ocean Protocol contributors
# SPDX-License-Identifier: Apache-2.0
#
# Usage: ./start_ocean.sh
#
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

IP="localhost"
optspec=":-:"
while getopts "$optspec" optchar; do
    case "${optchar}" in
           -)
           case "${OPTARG}" in
                exposeip)
                    IP="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
                 ;;
            esac;;
    esac
done

set -e

# Patch $DIR if spaces
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
DIR="${DIR/ /\\ }"
COMPOSE_DIR="${DIR}/barge/compose-files"
BACKEND_DIR="${DIR}/backend"


# Default versions of Aquarius, Provider

export AQUARIUS_VERSION=${AQUARIUS_VERSION:-v4.2.0}
export PROVIDER_VERSION=${PROVIDER_VERSION:-v1.3.2}
export SUBGRAPH_VERSION=${SUBGRAPH_VERSION:-v2.0.1}
export CONTRACTS_VERSION=${CONTRACTS_VERSION:-v1.0.0}
export RBAC_VERSION=${RBAC_VERSION:-next}

export OPERATOR_SERVICE_VERSION=${OPERATOR_SERVICE_VERSION:-oceanprotocol/operator-service:v4main}
export OPERATOR_ENGINE_VERSION=${OPERATOR_ENGINE_VERSION:-oceanprotocol/operator-engine:v4main}
export POD_CONFIGURATION_VERSION=${POD_CONFIGURATION_VERSION:-oceanprotocol/pod-configuration:v4main}
export POD_PUBLISHING_VERSION=${POD_PUBLISHING_VERSION:-oceanprotocol/pod-publishing:v4main}
export WAIT_FOR_C2DIMAGES=${WAIT_FOR_C2DIMAGES:-false}

export PROJECT_NAME="ocean"
export FORCEPULL="false"
export FORCEBUILD="false"

# Export User UID and GID
export LOCAL_USER_ID=$(id -u)
export LOCAL_GROUP_ID=$(id -g)



# Specify the ethereum default RPC container provider
if [ ${IP} = "localhost" ]; then
    export NETWORK_RPC_HOST="172.15.0.3"
else
    export NETWORK_RPC_HOST=${IP}
fi
export NETWORK_RPC_PORT="8545"
export NETWORK_RPC_URL="http://"${NETWORK_RPC_HOST}:${NETWORK_RPC_PORT}
export GANACHE_PORT="8545"
export GANACHE_HOST="172.15.0.3"
export GANACHE_URL="http://"${GANACHE_HOST}:${GANACHE_PORT}

# export NETWORK_RPC_URL='https://polygon-mumbai.g.alchemy.com/v2/YtTw29fEGWDXcMKpljSM63DbOrgXgJRx'
# Use this seed on ganache to always create the same wallets
export GANACHE_MNEMONIC=${GANACHE_MNEMONIC:-"taxi music thumb unique chat sand crew more leg another off lamp"}
export WEB3_INFURA_PROJECT_ID="4b1e6d019d6644de887db1255319eff8"
export WEB3_INFURA_URL=" https://mainnet.infura.io/v3/${WEB3_INFURA_PROJECT_ID}"
export WEB3_ALCHEMY_PROJECT_ID="RrtpZjiUVoViiDEaYxhN9o6m1CSIZvlL"
export WEB3_ALCHEMY_URL="https://eth-mainnet.g.alchemy.com/v2/${WEB3_INFURA_PROJECT_ID}"
# Ocean contracts

export PRIVATE_KEY="0x8467415bb2ba7c91084d932276214b11a3dd9bdb2930fefa194b666dd8020b99"
export OCEAN_HOME="${HOME}/.ocean"
export CONTRACTS_OWNER_ROLE_ADDRESS="${CONTRACTS_OWNER_ROLE_ADDRESS}"
export DEPLOY_CONTRACTS=true
export DEPLOY_SUBGRAPH=true
export OCEAN_ARTIFACTS_FOLDER="${OCEAN_HOME}/ocean-contracts/artifacts"

mkdir -p ${OCEAN_ARTIFACTS_FOLDER}
export OCEAN_C2D_FOLDER="${OCEAN_HOME}/ocean-c2d/"
mkdir -p ${OCEAN_C2D_FOLDER}
export ADDRESS_FILE="${OCEAN_ARTIFACTS_FOLDER}/address.json"
echo "export ADDRESS_FILE=${ADDRESS_FILE}"

#certs folder
export OCEAN_CERTS_FOLDER="${OCEAN_HOME}/ocean-certs/"
mkdir -p ${OCEAN_CERTS_FOLDER}
# copy certs
cp -r ./barge/certs/* ${OCEAN_CERTS_FOLDER}
# Specify which ethereum client to run or connect to: development
export CONTRACTS_NETWORK_NAME="development"

# Default Aquarius parameters: use Elasticsearch
export DB_MODULE="elasticsearch"
export DB_HOSTNAME="172.15.0.6"
export DB_PORT="9200"
export DB_USERNAME="elastic"
export DB_PASSWORD="changeme"
export DB_SSL="false"
export DB_VERIFY_CERTS="false"
export DB_CA_CERTS=""
export DB_CLIENT_KEY=""
export DB_CLIENT_CERT=""
CHECK_ELASTIC_VM_COUNT=true

export IPFS_GATEWAY=http://172.15.0.16:5001
export IPFS_HTTP_GATEWAY=http://172.15.0.16:8080
#Provider
export PROVIDER_LOG_LEVEL=${PROVIDER_LOG_LEVEL:-INFO}
export PROVIDER_WORKERS=4
# export PROVIDER_IPFS_GATEWAY=https://ipfs.oceanprotocol.com
export PROVIDER_IPFS_GATEWAY=${IPFS_HTTP_GATEWAY}
export PROVIDER_PRIVATE_KEY=0xfd5c1ccea015b6d663618850824154a3b3fb2882c46cefb05b9a93fea8c3d215

if [ ${IP} = "localhost" ]; then
    export AQUARIUS_URI=http://172.15.0.5:5000
else
    export AQUARIUS_URI=http://${IP}:5000
fi

#export OPERATOR_SERVICE_URL=http://127.0.0.1:8050
export OPERATOR_SERVICE_URL=${OPERATOR_SERVICE_URL:-"http://172.15.0.13:31000/"}

# Add aquarius to /etc/hosts
# Workaround mainly for macOS


# colors
COLOR_R="\033[0;31m"    # red
COLOR_G="\033[0;32m"    # green
COLOR_Y="\033[0;33m"    # yellow
COLOR_B="\033[0;34m"    # blue
COLOR_M="\033[0;35m"    # magenta
COLOR_C="\033[0;36m"    # cyan

# reset
COLOR_RESET="\033[00m"

function show_banner {
    local output=$(cat .banner)
    echo -e "$COLOR_B$output$COLOR_RESET"
    echo ""
}


function check_if_owned_by_root {
    if [ -d "$OCEAN_HOME" ]; then
        uid=$(ls -nd "$OCEAN_HOME" | awk '{print $3;}')
        if [ "$uid" = "0" ]; then
            printf $COLOR_R"WARN: $OCEAN_HOME is owned by root\n"$COLOR_RESET >&2
        else
            uid=$(ls -nd "$OCEAN_ARTIFACTS_FOLDER" | awk '{print $3;}')
            if [ "$uid" = "0" ]; then
                printf $COLOR_R"WARN: $OCEAN_ARTIFACTS_FOLDER is owned by root\n"$COLOR_RESET >&2
            fi
        fi
    fi
}


function check_max_map_count {
  vm_max_map_count=$(docker run --rm busybox sysctl -q vm.max_map_count)
  vm_max_map_count=${vm_max_map_count##* }
  vm_max_map_count=262144
  if [ $vm_max_map_count -lt 262144 ]; then
    printf $COLOR_R'vm.max_map_count current kernel value ($vm_max_map_count) is too low for Elasticsearch\n'$COLOR_RESET
    printf $COLOR_R'You must update vm.max_map_count to at least 262144\n'$COLOR_RESET
    printf $COLOR_R'Please refer to https://www.elastic.co/guide/en/elasticsearch/reference/6.6/vm-max-map-count.html\n'$COLOR_RESET
    exit 1
  fi
}

function clean_local_contracts {
    rm -f "${OCEAN_ARTIFACTS_FOLDER}/ready"
    rm -f "${OCEAN_ARTIFACTS_FOLDER}/*.json"
}

check_if_owned_by_root
show_banner

COMPOSE_FILES=""
# COMPOSE_FILES+=" -f ${DIR}/network/network_volumes.yml"

DOCKER_COMPOSE_EXTRA_OPTS="${DOCKER_COMPOSE_EXTRA_OPTS:-}"

while :; do
    case $1 in
        --exposeip)
	   ;;
        #################################################
        # Disable color
        #################################################

        --no-ansi)
            DOCKER_COMPOSE_EXTRA_OPTS+=" --no-ansi"
            ;;

        #################################################
        # Exclude switches
        #################################################
        --ganache)
            COMPOSE_FILES+=" -f ${DIR}/ganache/docker-compose.yml"
        ;;

        --contracts)
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/ocean_contracts.yml"
            ;;

        --c2d)
            
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/c2d.yml"

            printf $COLOR_Y'Starting with C2D...\n\n'$COLOR_RESET
            ;;
        --with-rbac)
	        COMPOSE_FILES+=" -f ${COMPOSE_DIR}/rbac.yml"
            printf $COLOR_Y'Starting with RBAC Server...\n\n'$COLOR_RESET
            ;;

        --backend)
	        COMPOSE_FILES+=" -f ${DIR}/backend/backend.yml"
            ;;


        --provider)
            COMPOSE_FILES+=" -f ${DIR}/provider/docker-compose.yml"
        ;;


        --ipfs)
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/ipfs.yml"


        ;;


        --frontend)
            COMPOSE_FILES+=" -f ${DIR}/frontend/docker-compose.yml"
        
            ;;
        --aquarius)
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/redis.yml"
            # COMPOSE_FILES+=" -f  ${DIR}/elasticsearch/elasticsearch.yml"
            # COMPOSE_FILES+=" -f ${DIR}/aquarius/docker-compose.yml"
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/elasticsearch.yml"
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/aquarius.yml"
            ;;
        --dashboard)
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/dashboard.yml"
            ;;
        --skip-deploy)
            export DEPLOY_CONTRACTS=false
            printf $COLOR_Y'Ocean contracts will not be deployed, the last deployment (if any) will be intact ...\n\n'$COLOR_RESET
            ;;
        #################################################
        # Cleaning switches
        #################################################
        --purge)
            eval docker-compose --project-name=$PROJECT_NAME "$COMPOSE_FILES" down;
            docker network rm ${PROJECT_NAME}_default || true;
            docker network rm ${PROJECT_NAME}_backend || true;
            shift
            break
            ;;

        --down)
        printf $COLOR_R'Restarting...\n\n'$COLOR_RESET
        eval docker-compose --project-name=$PROJECT_NAME "$COMPOSE_FILES" down;
        shift
        ;;
        --update)
            eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME "$COMPOSE_FILES" pull
            eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME "$COMPOSE_FILES" build
            shift
        ;;

        --build)
        
        eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME "$COMPOSE_FILES" build

        shift
        ;;
        --pull)
        
        eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME "$COMPOSE_FILES" pull

        shift
        ;;

        --restart)

            eval docker-compose --project-name=$PROJECT_NAME "$COMPOSE_FILES" restart;
            # docker network rm ${PROJECT_NAME}_default || true;
            # docker network rm ${PROJECT_NAME}_backend || true;


            shift
            break
            ;;


        --all)
        
            # backend
            COMPOSE_FILES+=" -f ${DIR}/backend/backend.yml"
            # deploy from barge
            COMPOSE_FILES+=" -f ${DIR}/network/network_volumes.yml"

            COMPOSE_FILES+=" -f ${DIR}/ganache/docker-compose.yml"
            # contracts
            # COMPOSE_FILES+=" -f ${DIR}/contracts/docker-compose.yml"
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/ocean_contracts.yml"
            # aquarius
            # COMPOSE_FILES+=" -f  ${DIR}/elasticsearch/elasticsearch.yml"
            # COMPOSE_FILES+=" -f ${DIR}/aquarius/docker-compose.yml"
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/aquarius.yml"
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/elasticsearch.yml"

            COMPOSE_FILES+=" -f ${DIR}/ipfs/ipfs.yml"
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/provider.yml"
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/redis.yml"
    `
            # provider
            # COMPOSE_FILES+=" -f ${DIR}/provider/docker-compose.yml"

            # COMPOSE_FILES+= 


            # COMPOSE_FILES+=
  

 

            # dashboard
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/dashboard.yml"`
        
        ;;



        --) # End of all options.
            shift
            break
            ;;

        --light)
            COMPOSE_FILES+=" -f ${DIR}/backend/backend.yml" 
            COMPOSE_FILES+=" -f ${DIR}/ganache/docker-compose.yml"
            COMPOSE_FILES+=" -f ${COMPOSE_DIR}/ipfs.yml"
            # shift
        ;;
        --up)
        [ ${CHECK_ELASTIC_VM_COUNT} = "true" ] && check_max_map_count
        eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME  "$COMPOSE_FILES" up  -d 
        break
        ;;
        -?*)
            printf $COLOR_R'WARN: Unknown option (ignored): %s\n'$COLOR_RESET "$1" >&2
            break
            ;;
        *)
            # [ ${CHECK_ELASTIC_VM_COUNT} = "true" ] && check_max_map_count
            # eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME  "$COMPOSE_FILES" up --remove-orphans -d 
            # break
    esac
    shift
done



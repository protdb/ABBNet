#!/bin/bash
# Container and image base names
CONTAINER_BASE_NAME="abbnet"
IMAGE_BASE_NAME="abbnet_search"

declare -A DOCKER_PARAMS
# launch parameters for every environment
DOCKER_PARAMS["localhost"]=""

declare -A VOLUMES
# volume setup for every environment
VOLUMES["localhost"]="-v /home/gluck/abb_data:/data -v /home/gluck/pdbs:/pdb"

declare -A COMMANDS
# available commands list
# if passed, overrides command in Dockerfile's CMD
COMMANDS["bash"]="bash"
#!/bin/bash
source run_params.sh
REBUILD=1
COMMAND=''
DETACH='--rm'
ENVIRONMENT='localhost'
CONFIG=""
DOCKER_COMMAND="docker run"
CONTAINER_NAME=""
VOLUME=""

while getopts c:e:f:n:g:v:dikh flag
do
  case "${flag}" in
    f) CONFIG=${OPTARG};; # Config file
    k) REBUILD=0;; # keep (from rebuild)
    c) COMMAND=${COMMANDS[${OPTARG}]};; #command from COMMANDS in run_params.sh. if not specified, no command is passed to container
    e) ENVIRONMENT=${OPTARG};;
    d) DETACH='-d --restart=always';; # if not set, docker runs with --rm parameter, if set - with -d parameter (set -d in production)
    i) DETACH="--rm -ti";;
    n) CONTAINER_NAME=${OPTARG};;
    g) DOCKER_COMMAND="nvidia-docker run --gpus='\"device=${OPTARG}\"'";;
    h) echo "Common service runner script, configured to run container named $CONTAINER_BASE_NAME from image $IMAGE_BASE_NAME;
      options:
      -e: environment identifier
      -g <gpu_id>: If set, uses nvidia-docker instead of docker to run the worker with given GPU ID
      -f <config file>: override environment file, default is 'env/{environment}.env'
      -s <standalone_message_file>: if set, env var STANDALONE_MESSAGE will be passed to container overriding .env file
      -k: keep existing image. If set, script will not rebuild container
      -i: interactive mode
      -c <command>: specifies the command to run in container. Commands must be listed in run_params.sh as elem of array COMMANDS. Default is no command provided (container will run wih CMD option from Dockerfile)
      -d: container will run as detached and will not be removed after stop (default will add --rm to docker run, if specified there will be -d option added
      -n: overwrite default container name configured as ${CONTAINER_BASE_NAME}_{ENVIRONMENT} to avoid conflicts with working container
      -v <directory>: overrides volume mounting
      -h: shows this help and exit.
      Common usages:
      run in prod: ./run.sh -de prod
      enter interactive mode: ./run.sh -ikc bash
      run with environment run_params using separate env-file:
          ./run.sh -e standalone -f env/override_params.env
      run using GPU:
         ./run.sh -e test -g 1
      "
      exit 0;;
    *) echo "Error: unknown parameter ${flag}. use run.sh -h to get help"
       exit 1
  esac
done
if [[ $CONFIG == "" ]]; then
  CONFIG="env/${ENVIRONMENT}.env"
fi
if [[ $CONTAINER_NAME == "" ]]; then
  CONTAINER_NAME="${CONTAINER_BASE_NAME}_${ENVIRONMENT}"
fi
IMAGE_NAME="${IMAGE_BASE_NAME}_${ENVIRONMENT}"
DOCKER_OPTS="${DOCKER_PARAMS[$ENVIRONMENT]} ${VOLUMES[$ENVIRONMENT]}"

docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"
if [[ $REBUILD == 1 ]]; then
  docker build -t "$IMAGE_NAME" .
fi
eval "$DOCKER_COMMAND $DETACH --network=host --name $CONTAINER_NAME $DOCKER_OPTS --tmpfs=/ramdisk --env-file=$CONFIG $IMAGE_NAME $COMMAND"


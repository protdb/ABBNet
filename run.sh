#!/bin/bash
source run_params.sh
REBUILD=1
COMMAND=''
DETACH='--rm'
ENVIRONMENT='localhost'
CONFIG=""
DOCKER_COMMAND="docker run"
CONTAINER_NAME=""
CACHE=""
VOLUME=""

while getopts c:e:f:n:g:v:dipkh flag
do
  case "${flag}" in
    f) CONFIG=${OPTARG};; # Config file
    k) REBUILD=0;; # keep (from rebuild)
    c) COMMAND=${COMMANDS[${OPTARG}]};; #command from COMMANDS in run_params.sh. if not specified, no command is passed to container
    e) ENVIRONMENT=${OPTARG};;
    d) DETACH='-d --restart=always';; # if not set, docker runs with --rm parameter, if set - with -d parameter (set -d in production)
    i) DETACH="--rm -ti";;
    n) CONTAINER_NAME=${OPTARG};;
    p) CACHE=" --no-cache";;
    v) VOLUME="-v ${OPTARG}:/data";;
    g) DOCKER_COMMAND="nvidia-docker run --gpus='\"device=${OPTARG}\"'";;
    h) echo "Common service runner script, configured to run container named $CONTAINER_BASE_NAME from image $IMAGE_BASE_NAME;
      options:
      MOST COMMON:
      -e <environment>: environment identifier. Environment uses env/{environment}.env file as --env-file and adds volumes and options described for this environment in run_params.sh. Environment name will be added to container and image names
      -d: detached mode. Container will run as detached and will not be removed after stop (default will add --rm to docker run; specifying -f flag here will replace it with -d --restart=always)
      -g <gpu_id>: If set, uses nvidia-docker instead of docker to run the worker with given GPU ID
      -k: keep existing image. If set, script will not rebuild container
      SPECIFIC CASES:
      -c <command>: specifies the command to run in container. Commands must be listed in run_params.sh as elem of array COMMANDS. Default is no command provided (container will run wih CMD option from Dockerfile)
      -f <config file>: override environment file, default is 'env/{environment}.env'
      -i: interactive mode (for commands supporting user interactions, like running bash in the container)
      -n: overwrite default container name configured as ${CONTAINER_BASE_NAME}_{ENVIRONMENT} to avoid conflicts with working container
      -v <directory>: overrides ALL volume mounting configured in run_params.sh and mounts given path to /data
      -p: pure build (adds --no-cache to docker build command)
      -h: shows this help and exit.
      Common usages:
      run in prod: ./run.sh -de prod
      run in prod using GPU with id 0: ./run.sh -de prod -g 0
      enter interactive mode: ./run.sh -ikc bash (if bash is specified as command in run_params.sh
      run using separate env-file:
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
if [[ $VOLUME == "" ]]; then
  VOLUME=${VOLUMES[$ENVIRONMENT]}
fi
IMAGE_NAME="${IMAGE_BASE_NAME}_${ENVIRONMENT}"
DOCKER_OPTS="${DOCKER_PARAMS[$ENVIRONMENT]} $VOLUME"

docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"
if [[ $REBUILD == 1 ]]; then
  docker build -t "$IMAGE_NAME" .${CACHE}
fi
eval "$DOCKER_COMMAND $DETACH --network=host --name $CONTAINER_NAME $DOCKER_OPTS --tmpfs=/ramdisk --env-file=$CONFIG $IMAGE_NAME $COMMAND"


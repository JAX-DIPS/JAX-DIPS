#!/bin/bash
###############################################################################
#
# This is my $LOCAL_ENV file
#
LOCAL_ENV=.env
#
###############################################################################

usage() {
    cat <<EOF

USAGE: launch.sh

launch utility script
----------------------------------------

launch.sh [command]

    valid commands:

    build
    pull
    push
    dev
    root
    jupyter


Getting Started tl;dr
----------------------------------------

    ./launch.sh build
    ./launch.sh dev
For more detailed info on getting started, see README.md


More Information
----------------------------------------

Note: This script looks for a file called $LOCAL_ENV in the
current directory. This file should define the following environment
variables:
    JUPYTER_PORT
        Port for launching jupyter lab, e.g. 8888
    DATA_PATH
        path to data directory. e.g., /scratch/data
    DATA_MOUNT_PATH
        Path to data inside container. e.g., /data
    REGISTRY
        container registry URL. e.g., nvcr.io. Only required to push/pull containers.
    REGISTRY_USER
        container registry username. e.g., '$oauthtoken' for registry access. Only required to push/pull containers.
    REGISTRY_ACCESS_TOKEN
        container registry access token. e.g., Ckj53jGK... Only required to push/pull containers.
    WANDB_API_KEY
        Weights and Balances API key to upload runs to WandB. Can also be uploaded afterwards., e.g. Dkjdf...
        This value is optional -- Weights and Biases will log data and not upload if missing.

EOF
    exit
}

CONT_NAME=${CONT_NAME:=jax_dips}
IMAGE_NAME=${IMAGE_NAME:=docker.io/pourion/jax_dips:latest}
REGISTRY_USER=${REGISTRY_USER:='$oauthtoken'}
REGISTRY=${REGISTRY:=NotSpecified}
REGISTRY_ACCESS_TOKEN=${REGISTRY_ACCESS_TOKEN:=NotSpecified}

###############################################################################
#
# if $LOCAL_ENV file exists, source it to specify my environment
#
###############################################################################

if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
    write_env=0
else
    echo $LOCAL_ENV does not exist. Writing deafults to $LOCAL_ENV
    write_env=1
fi

###############################################################################
#
# If $LOCAL_ENV was not found, write out a template for user to edit
#
###############################################################################

if [ $write_env -eq 1 ]; then
    echo CONT_NAME=${CONT_NAME} >> $LOCAL_ENV
    echo IMAGE_NAME=${IMAGE_NAME} >> $LOCAL_ENV
    echo JUPYTER_PORT=${JUPYTER_PORT} >> $LOCAL_ENV
    echo DATA_PATH=${DATA_PATH} >> $LOCAL_ENV
    echo DATA_MOUNT_PATH=${DATA_MOUNT_PATH} >> $LOCAL_ENV
    echo RESULT_MOUNT_PATH=${RESULT_MOUNT_PATH} >> $LOCAL_ENV
    echo RESULT_PATH=${RESULT_PATH} >> $LOCAL_ENV
    echo REGISTRY_USER=${REGISTRY_USER} >> $LOCAL_ENV
    echo REGISTRY=${REGISTRY} >> $LOCAL_ENV
    echo REGISTRY_ACCESS_TOKEN=${REGISTRY_ACCESS_TOKEN} >> $LOCAL_ENV
    echo WANDB_API_KEY=${WANDB_API_KEY} >> $LOCAL_ENV
fi

###############################################################################

DOCKER_CMD="docker run \
    --network host \
    --gpus all \
    -p ${JUPYTER_PORT}:8888 \
    -v ${DATA_PATH}:${DATA_MOUNT_PATH} \
    -v ${RESULT_PATH}:${RESULT_MOUNT_PATH} \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -v $(pwd):/workspace \
    -e HOME=/workspace \
    -w /workspace"


build() {
    set -x
    local IMG_BASENAME=($(echo ${IMAGE_NAME} | tr ":" "\n"))
    DOCKER_FILE="docker/Dockerfile"

    echo -e "Building ${DOCKER_FILE}..."
    docker build --network host --ssh default \
        -t ${IMAGE_NAME} \
        -t ${IMG_BASENAME[0]}:latest \
        -f ${DOCKER_FILE} .
}


dev() {
    local DEV_IMG=${IMAGE_NAME}
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--image)
                DEV_IMG="$2"
                shift
                shift
                ;;
	    -d|--deamon)
                DOCKER_CMD="${DOCKER_CMD} -d"
                shift
                ;;
            *)
                echo "Unknown option $1"
                exit 1
                ;;
        esac
    done
    DEV_PYTHONPATH='/workspace:/opt/PyEVTK'
    $DOCKER_CMD \
        --name ${CONT_NAME} \
        --env WANDB_API_KEY=$WANDB_API_KEY \
        -u $(id -u):$(id -u) \
        -e PYTHONPATH=$DEV_PYTHONPATH \
        -it --rm \
        ${DEV_IMG} \
        bash
}


push() {
    local IMG_BASENAME=($(echo ${IMAGE_NAME} | tr ":" "\n"))
    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    # docker push ${IMG_BASENAME[0]}:latest
    docker push ${IMAGE_NAME}
    exit
}


pull() {
    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker pull ${IMAGE_NAME}
    exit
}


attach() {
    DOCKER_CMD="docker exec"
    CONTAINER_ID=$(docker ps | grep ${CONT_NAME} | cut -d' ' -f1)
    ${DOCKER_CMD} -it ${CONTAINER_ID} /bin/bash
    exit
}


root() {
    ${DOCKER_CMD} -it --user root ${IMAGE_NAME} bash
    exit
}


jupyter() {
    ${DOCKER_CMD} -it ${IMAGE_NAME} jupyter-lab --no-browser \
        --port=${JUPYTER_PORT} \
        --ip=0.0.0.0 \
        --allow-root \
        --notebook-dir=/workspace \
        --NotebookApp.password='' \
        --NotebookApp.token='' \
        --NotebookApp.password_required=False
}


case $1 in
    build)
        ;&
    push)
        ;&
    pull)
        ;&
    dev)
        $@
        ;;
    attach)
        $@
        ;;
    root)
        ;&
    jupyter)
        $1
        ;;
    *)
        usage
        ;;
esac
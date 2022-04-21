# JAX-DIPS
JAX implementation of a differentiable inverse PDE solver with irregular interface.

![me](https://github.com/JAX-DIPS/JAX-DIPS/blob/main/sample_adv_semi_lagrangian.gif)


# Pre-req

## Nvidia Driver

```
apt install $(nvidia-detector)
```

## Docker Engine

Please refer https://docs.docker.com/engine/install/ubuntu/

## nvidia-docker2

Please refer https://nvidia.github.io/nvidia-docker/ to setup apt repo

apt-get install nvidia-docker2


# Build development container

```
./launch build
```

# Start developement container
This will create a container and places the user in the container with source code mounted.

Once the container is created, user can attach to this container from VS code.

```
./launch dev
```
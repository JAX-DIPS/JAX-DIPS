import numpy
import torch
import warp
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack


def jax_to_torch(x_jax):
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch


def torch_to_jax(x_torch):
    x_torch = x_torch.contiguous()
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax


# wrap a torch tensor to a wp array, data is not copied
def torch_to_warp(t, dtype=warp.types.float32):
    # ensure tensors are contiguous
    assert t.is_contiguous()

    rows = 0

    if len(t.shape) > 1 and warp.type_length(dtype) == 1:
        rows = t.numel()
    elif len(t.shape) == 1:
        rows = t.shape[0]

    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError("Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type")

    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=dtype,
        length=rows,
        copy=False,
        owner=False,
        requires_grad=True,
        device=t.device.type,
    )

    # save a reference to the source tensor, otherwise it will be deallocated
    a.tensor = t

    return a


def warp_to_torch(a):
    if a.device == "cpu":
        # Torch has an issue wrapping CPU objects
        # that support the __array_interface__ protocol
        # in this case we need to workaround by going
        # to an ndarray first, see https://pearu.github.io/array_interface_pytorch.html
        return torch.as_tensor(numpy.asarray(a))

    elif a.device == "cuda":
        # Torch does support the __cuda_array_interface__
        # correctly, but we must be sure to maintain a reference
        # to the owning object to prevent memory allocs going out of schope
        return torch.as_tensor(a, device="cuda")

    else:
        raise RuntimeError("Unsupported device")


def warp_to_jax(x_warp):
    x_torch = warp_to_torch(x_warp)
    return torch_to_jax(x_torch)


def jax_to_warp(x_jax):
    x_torch = jax_to_torch(x_jax)
    return torch_to_warp(x_torch)


if __name__ == "__main__":
    import pdb

    import jax.numpy as jnp

    warp.init()

    aa = jnp.zeros(10, dtype=float)
    bb = jax_to_torch(aa)
    cc = torch_to_warp(bb)

    print(cc.numpy())
    pdb.set_trace()

from jax_dips.utils.conversions import torch_to_jax
from .inn_e4.GenerateData import Data
from .inn_e4.Tool import data_transform

import torch
import numpy as np
from jax import numpy as jnp

alpha = 7.0465 * 1e3
kbT = 0.592783
pi = torch.tensor(np.pi)
cs = 80
cp = 2
Is = 1e-5
k2 = 8.4869 * Is / cs
k = float(np.sqrt(k2))


def g(x):
    x_norm = torch.norm(x, dim=1)
    return (alpha / (4 * pi * cp * x_norm)).view(-1, 1)


def true_solution_net(x, r0, label="inner", device="cpu"):
    x_norm = torch.norm(x, dim=1)
    r0 = torch.tensor(r0).to(device)
    if label == "inner":
        shape = torch.ones_like(x_norm).to(device)
        return (alpha / (4 * pi * r0) * (1 / (cs * (1 + k * r0)) - 1 / cp) * shape).view(-1, 1)
    elif label == "out":
        return (alpha / (4 * pi * cs) * torch.exp(k * r0) / (1 + k * r0) * torch.exp(-k * x_norm) / x_norm).view(-1, 1)


def inner_or_not(input, args):
    """
    Omega_1 or Omega_2
    """
    X_norm = np.linalg.norm(input - args.L, axis=1, ord=2)
    index_inner = np.where(X_norm < args.r0)[0]
    index_out = np.where(X_norm >= args.r0)[0]

    return index_inner, index_out


def test_data_net(device, args):
    r0 = args.r0
    h = 0.125
    X = np.arange(-2.5, 2.5 + h, h)
    Y = np.arange(-2.5, 2.5 + h, h)
    Z = np.arange(-2.5, 2.5 + h, h)
    X, Y, Z = np.meshgrid(X, Y, Z)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    Z = Z.reshape(-1, 1)
    input = np.hstack((X, Y, Z))
    index_inner, index_out = inner_or_not(input, args)
    input = torch.tensor(input).float()
    inner = input[index_inner, :]
    test_inner = inner.float().to(device).clone().detach()
    label_inner = true_solution_net(test_inner, r0, "inner", device).clone().detach()

    out = input[index_out, :]
    test_out = out.float().to(device).clone().detach()
    label_out = true_solution_net(test_out, r0, "out", device).clone().detach()

    return test_out, label_out, test_inner, label_inner


class INNSingleSphereData:
    def __init__(
        self,
        sigma=1.0,  # sphere radius
        L=[0, 0, 0],  # sphere center coord
        box=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],  # box dimensions
        device="cuda",
        train_out=2000,  # points outside/positive domain
        train_inner=100,  # points inside/negative domain
        train_boundary=1000,  # points on the boundary
        train_gamma=200,  # points on interface
    ):
        data = Data(r0=sigma, L=L, box=box, device=device)
        # outside region points
        out = data.SampleFromOut(train_out).T
        x_out, y_out, z_out, input_out = data_transform(out)
        self.x_out = jnp.squeeze(torch_to_jax(x_out))
        self.y_out = jnp.squeeze(torch_to_jax(y_out))
        self.z_out = jnp.squeeze(torch_to_jax(z_out))
        # inside region points
        inner = data.SampleFromInner(train_inner).T
        x_in, y_in, z_in, input_in = data_transform(inner)
        self.x_in = jnp.squeeze(torch_to_jax(x_in))
        self.y_in = jnp.squeeze(torch_to_jax(y_in))
        self.z_in = jnp.squeeze(torch_to_jax(z_in))
        # interface points
        gamma = data.SampleFromGamma(train_gamma).T
        x_gamma, y_gamma, z_gamma, input_in_b = data_transform(gamma)
        self.x_gamma = jnp.squeeze(torch_to_jax(x_gamma))
        self.y_gamma = jnp.squeeze(torch_to_jax(y_gamma))
        self.z_gamma = jnp.squeeze(torch_to_jax(z_gamma))
        # box boundaries
        input_boundary = data.SampleFromBoundary(train_boundary)
        input_boundary_label = true_solution_net(input_boundary, sigma, "out", device)
        self.input_boundary = jnp.squeeze(torch_to_jax(input_boundary))
        self.input_boundary_label = jnp.squeeze(torch_to_jax(input_boundary_label))
        self.R_xmin = jnp.squeeze(torch_to_jax(data.P_xmin))
        self.R_xmax = jnp.squeeze(torch_to_jax(data.P_xmax))
        self.R_ymin = jnp.squeeze(torch_to_jax(data.P_ymin))
        self.R_ymax = jnp.squeeze(torch_to_jax(data.P_ymax))
        self.R_zmin = jnp.squeeze(torch_to_jax(data.P_zmin))
        self.R_zmax = jnp.squeeze(torch_to_jax(data.P_zmax))

        # aggregated coordinates
        self.x = jnp.concatenate((self.x_in, self.x_gamma, self.x_out, self.input_boundary[:, 0]))
        self.y = jnp.concatenate((self.y_in, self.y_gamma, self.y_out, self.input_boundary[:, 1]))
        self.z = jnp.concatenate((self.z_in, self.z_gamma, self.z_out, self.input_boundary[:, 2]))


class INNDoubleSphereData:
    def __init__(
        self,
        sigma=1.0,  # sphere radius
        L=[0, 0, 0],  # sphere center coord
        box=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],  # box dimensions
        device="cuda",
        train_out=2000,  # points outside/positive domain
        train_inner=100,  # points inside/negative domain
        train_boundary=1000,  # points on the boundary
        train_gamma=200,  # points on interface
    ):
        data = Data(r0=sigma, L=L, box=box, device=device)
        # outside region points
        out = data.SampleFromOut(train_out).T
        x_out, y_out, z_out, input_out = data_transform(out)
        self.x_out = jnp.squeeze(torch_to_jax(x_out))
        self.y_out = jnp.squeeze(torch_to_jax(y_out))
        self.z_out = jnp.squeeze(torch_to_jax(z_out))
        # inside region points
        inner = data.SampleFromInner(train_inner).T
        x_in, y_in, z_in, input_in = data_transform(inner)
        self.x_in = jnp.squeeze(torch_to_jax(x_in))
        self.y_in = jnp.squeeze(torch_to_jax(y_in))
        self.z_in = jnp.squeeze(torch_to_jax(z_in))
        # interface points
        gamma = data.SampleFromGamma(train_gamma).T
        x_gamma, y_gamma, z_gamma, input_in_b = data_transform(gamma)
        self.x_gamma = jnp.squeeze(torch_to_jax(x_gamma))
        self.y_gamma = jnp.squeeze(torch_to_jax(y_gamma))
        self.z_gamma = jnp.squeeze(torch_to_jax(z_gamma))
        # box boundaries
        input_boundary = data.SampleFromBoundary(train_boundary)
        input_boundary_label = true_solution_net(input_boundary, sigma, "out", device)
        self.input_boundary = jnp.squeeze(torch_to_jax(input_boundary))
        self.input_boundary_label = jnp.squeeze(torch_to_jax(input_boundary_label))
        self.R_xmin = jnp.squeeze(torch_to_jax(data.P_xmin))
        self.R_xmax = jnp.squeeze(torch_to_jax(data.P_xmax))
        self.R_ymin = jnp.squeeze(torch_to_jax(data.P_ymin))
        self.R_ymax = jnp.squeeze(torch_to_jax(data.P_ymax))
        self.R_zmin = jnp.squeeze(torch_to_jax(data.P_zmin))
        self.R_zmax = jnp.squeeze(torch_to_jax(data.P_zmax))

        # aggregated coordinates
        self.x = jnp.concatenate((self.x_in, self.x_gamma, self.x_out, self.input_boundary[:, 0]))
        self.y = jnp.concatenate((self.y_in, self.y_gamma, self.y_out, self.input_boundary[:, 1]))
        self.z = jnp.concatenate((self.z_in, self.z_gamma, self.z_out, self.input_boundary[:, 2]))

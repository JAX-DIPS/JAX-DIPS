from functools import partial

import kaolin
import numpy as onp
import torch
from jax import jit
from jax import numpy as jnp
from jax import random, vmap

from jax_dips._jaxmd_modules.util import f32, i32
from jax_dips.domain import mesh
from jax_dips.utils.conversions import jax_to_torch, torch_to_jax


class DatasetDictMGPU:
    def __init__(self, x_data, batch_size=64, drop_remainder: bool = False, shuffle: bool = False):
        self.batch_size = batch_size
        self.x_data = x_data
        self._len = None
        self.drop_remainder = drop_remainder
        self.shuffle = shuffle

    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_data)
        self._order = onp.arange(self._len)
        if self.shuffle:
            onp.random.shuffle(self._order)
        return self

    def __next__(self):
        max_idx = self._len
        if self.drop_remainder:
            max_idx -= self.batch_size

        if self._idx >= max_idx:
            raise StopIteration

        data_x = {}
        batch_idx = self._order[self._idx : min(self._len, self._idx + self.batch_size)]

        data_x = self.x_data[batch_idx]
        self._idx += self.batch_size
        return data_x


class DatasetDict_:
    def __init__(self, x_data, batch_size):
        self.x_data = x_data
        self._len = len(x_data)
        self.batch_size = batch_size
        self.num_batches = self._len // self.batch_size

        self.gpu_data = x_data.reshape((self.num_batches, self.batch_size, -1))
        self._batch_counter = 0

    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_data)
        return self

    def __next__(self):
        if self._idx >= self._len:
            raise StopIteration
        self._idx += self.batch_size
        self._batch_counter += 1
        return self.gpu_data[self._batch_counter - 1]


def generate_random_points_like(x_data, num_missing_points):
    key = random.PRNGKey(0)
    cov = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mean = jnp.array([0.0, 0.0, 0.0])
    dR = random.multivariate_normal(key, mean, cov, shape=(num_missing_points,))
    Rnew = x_data.min() + (x_data.max() - x_data.min()) * (dR - dR.min()) / (dR.max() - dR.min())
    return Rnew


class DatasetDict:
    def __init__(self, x_data, batch_size, num_gpus=1):
        self.x_data = x_data

        self._len = len(x_data)
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        self._len_per_gpu = int(
            onp.ceil(self._len / self.num_gpus)
        )  # self._len // self.num_gpus + int(jnp.heaviside(self._len % self.num_gpus, 0))
        # self.extra_points_per_gpu = self._len_per_gpu * self.num_gpus - self._len

        if self.batch_size > self._len_per_gpu:
            self.batch_size = self._len_per_gpu

        self.extra_batch_per_gpu = int(jnp.heaviside(self._len_per_gpu % self.batch_size, 0))
        self.num_batches_per_gpu = self._len_per_gpu // self.batch_size
        self.gpu_padded_batches()

        if self.num_gpus == 1:
            self.batched_data = self.batched_data.reshape(self.batched_data.shape[1:])

        self._batch_counter = 0

    def gpu_padded_batches(self):
        """
        Pads the data in case it cannot be just folded exactly.
        The padded values are randomly scattered points inside the domain.
        dx, dy, dz are used to ensure
        """
        self.last_batch_size = self._len_per_gpu % self.batch_size

        self.batched_data = jnp.zeros(
            (
                self.num_gpus,
                self.num_batches_per_gpu + self.extra_batch_per_gpu,
                self.batch_size,
                3,
            )
        )

        for gpu in range(self.num_gpus):
            for batch_id in range(self.num_batches_per_gpu):
                data_begin_idx = (
                    gpu
                    * (self.num_batches_per_gpu * self.batch_size + self.extra_batch_per_gpu * self.last_batch_size)
                    + batch_id * self.batch_size
                )
                data_end_idx = min(data_begin_idx + self.batch_size, self._len)

                data_slice = self.x_data[data_begin_idx:data_end_idx]
                if data_slice.shape[0] < self.batch_size:
                    # Readjust data_slice's size to make sure merging it to
                    # batched_data does not fail due to shape mismatch
                    padding = generate_random_points_like(self.x_data, self.batch_size - data_slice.shape[0])
                    # TODO: Revisit to replace concatenate with writing in 2 steps
                    data_slice = jnp.concatenate((data_slice, padding))

                self.batched_data = self.batched_data.at[gpu, batch_id].set(data_slice)

            # TODO: This may not be required. Talk to Pouria to understand how
            #       block is used. Ideally the last batch will be of different
            #       size the that can be addressed in the previous block.
            if self.extra_batch_per_gpu == 1:
                data_begin_idx = data_end_idx  # gpu * (self.num_batches_per_gpu * self.batch_size + self.extra_batch_per_gpu * self.last_batch_size) + self.num_batches_per_gpu*self.batch_size
                if gpu == self.num_gpus - 1:
                    data_end_idx = self._len
                else:
                    data_end_idx = data_begin_idx + self.last_batch_size
                data_slice = self.x_data[data_begin_idx:data_end_idx]

                self.batched_data = self.batched_data.at[
                    gpu, self.num_batches_per_gpu, 0 : data_end_idx - data_begin_idx, :
                ].set(data_slice)

                num_missing_points = self.batch_size - (data_end_idx - data_begin_idx)
                Rnew = generate_random_points_like(self.x_data, num_missing_points)
                self.batched_data = self.batched_data.at[gpu, self.num_batches_per_gpu, -num_missing_points:, :].set(
                    Rnew
                )

    def get_batched_data(self):
        return self.batched_data

    def split_over_devices(self):
        import jax

        def split(arr):
            """
            Splits the second axis of `self.batched_data` evenly across the number of devices.
            first axis is the mini-batch dimension, second is the gpu device, third & fourth are the points.
            """
            return arr.reshape(arr.shape[0] // self.num_gpus, self.num_gpus, *arr.shape[1:])

        self.batched_data = jax.tree_map(split, self.batched_data)

    def __iter__(self):
        self._idx = 0
        self._len = len(self.x_data)
        return self

    def __next__(self):
        if self._idx >= self._len:
            raise StopIteration
        self._idx += self.batch_size
        self._batch_counter += 1
        return self.batched_data[self._batch_counter - 1]


class TrainData:
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, Nx=64, Ny=64, Nz=64) -> None:
        """Training GSTATE"""
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
        yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
        zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
        init_mesh_fn, coord_at = mesh.construct(3)
        self.gstate = init_mesh_fn(xc, yc, zc)

        self.key = random.PRNGKey(0)
        Lx = self.gstate.xmax() - self.gstate.xmin()
        Ly = self.gstate.ymax() - self.gstate.ymin()
        Lz = self.gstate.zmax() - self.gstate.zmin()
        self.LL = jnp.array([[Lx, Ly, Lz]])

        self.alt_res = False
        self.base_level = int(onp.log2(Nx))

        self.boundary_points = jnp.concatenate(
            (
                self.gstate.R_xmin_boundary,
                self.gstate.R_xmax_boundary,
                self.gstate.R_ymin_boundary,
                self.gstate.R_ymax_boundary,
                self.gstate.R_zmin_boundary,
                self.gstate.R_zmax_boundary,
            )
        )

    @partial(jit, static_argnums=(0))
    def move_train_points(self, points, dx, dy, dz):
        cov = jnp.array([[dx, 0.0, 0.0], [0.0, dy, 0.0], [0.0, 0.0, dz]]) * 0.5
        mean = jnp.array([0.0, 0.0, 0.0])
        Rnew = points + random.multivariate_normal(self.key, mean, cov, shape=(len(points),))
        new_points = Rnew - jnp.floor(Rnew / self.LL) * self.LL - 0.5 * self.LL
        return new_points

    @partial(jit, static_argnums=(0))
    def alternate_res(self, epoch, train_dx, train_dy, train_dz):
        self.alt_res = True
        train_dx = jnp.where(epoch % 4 == 0, self.gstate.dx, train_dx * 0.50)
        train_dy = jnp.where(epoch % 4 == 0, self.gstate.dy, train_dy * 0.50)
        train_dz = jnp.where(epoch % 4 == 0, self.gstate.dz, train_dz * 0.50)
        return train_dx, train_dy, train_dz

    @partial(jit, static_argnums=(0))
    def alternate_res_sequentially(self, num_epochs, epoch, train_dx, train_dy, train_dz):
        self.alt_res = False
        zoom_lvl = epoch // (num_epochs // 4)
        train_dx = self.gstate.dx * 0.5**zoom_lvl
        train_dy = self.gstate.dy * 0.5**zoom_lvl
        train_dz = self.gstate.dz * 0.5**zoom_lvl
        return train_dx, train_dy, train_dz

    def plot_slice(self, base_points):
        import matplotlib

        matplotlib.use("Agg")
        import os

        import matplotlib.pyplot as plt

        points = base_points.reshape((self.Nx, self.Ny, self.Nz, 3))
        currDir = os.path.dirname(os.path.realpath(__file__))
        rootDir = os.path.abspath(os.path.join(currDir, ".."))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            points[:, :, self.Nz // 2, 0],
            points[:, :, self.Nz // 2, 1],
            points[:, :, self.Nz // 2, 2],
            color="k",
        )
        filename = os.path.join(rootDir, "tests/grid.png")
        plt.savefig(filename)
        plt.close()

    def refine_normals(self, phi_fn_uns, max_iters=20):
        from jax import jit, vmap

        def phi_fn(r):
            return phi_fn_uns(jnp.squeeze(r))

        def normal_fn(point, dx, dy, dz):
            """
            Evaluate normal vector at a given point based on interpolated values
            of the level set function at the face-centers of a 3D cell centered at the
            point with each side length given by dx, dy, dz.
            """
            point_ip1_j_k = jnp.array([[point[0] + dx, point[1], point[2]]])
            point_im1_j_k = jnp.array([[point[0] - dx, point[1], point[2]]])
            phi_x = (phi_fn(point_ip1_j_k) - phi_fn(point_im1_j_k)) / (2 * dx)

            point_i_jp1_k = jnp.array([[point[0], point[1] + dy, point[2]]])
            point_i_jm1_k = jnp.array([[point[0], point[1] - dy, point[2]]])
            phi_y = (phi_fn(point_i_jp1_k) - phi_fn(point_i_jm1_k)) / (2 * dy)

            point_i_j_kp1 = jnp.array([[point[0], point[1], point[2] + dz]])
            point_i_j_km1 = jnp.array([[point[0], point[1], point[2] - dz]])
            phi_z = (phi_fn(point_i_j_kp1) - phi_fn(point_i_j_km1)) / (2 * dz)

            norm = jnp.sqrt(phi_x * phi_x + phi_y * phi_y + phi_z * phi_z)
            return jnp.array([phi_x / norm, phi_y / norm, phi_z / norm])

        base_points = self.gstate.R
        dx = self.gstate.dx
        dy = self.gstate.dy
        dz = self.gstate.dz

        vphi_fn = jit(vmap(phi_fn))
        vnormal_fn = jit(vmap(normal_fn, (0, None, None, None)))

        dt = 0.1
        max_disp = 0.0
        counter = 0
        while counter < max_iters:
            counter += 1
            arr_normal = vnormal_fn(base_points, dx, dy, dz)
            arr_phi = vphi_fn(base_points)
            displacement = -arr_phi[..., jnp.newaxis] * arr_normal * dt
            base_points += displacement
            max_disp += jnp.max(jnp.linalg.norm(displacement, axis=1))

        # self.plot_slice(base_points)
        return base_points

    def refine_LOD(self, phi_fn_uns, init_res=4, upsamples=6):
        def phi_fn(r):
            return phi_fn_uns(jnp.squeeze(r))

        def torch_phi_fn(points):
            phis = vmap(phi_fn)(torch_to_jax(points))
            return jax_to_torch(phis)

        Lx = float(self.LL.max())  # xmax - xmin
        xmin = -Lx / 2.0
        nx = init_res * 2**upsamples
        dx = Lx / nx

        binary_voxelgrid = kaolin.ops.conversions.sdf_to_voxelgrids(
            [torch_phi_fn],
            init_res=init_res,
            bbox_center=0.0,
            bbox_dim=Lx,
            upsampling_steps=upsamples,
        )

        bool_surface_voxels = kaolin.ops.voxelgrid.extract_surface(binary_voxelgrid, mode="wide")

        surface_voxels = torch.tensor(bool_surface_voxels, dtype=torch.uint8)
        ijk = jnp.where(torch_to_jax(surface_voxels[0]) > 0)

        # ijk = jnp.where(torch_to_jax(binary_voxelgrid[0])>0)
        xp = xmin + ijk[0] * dx
        yp = xmin + ijk[1] * dx
        zp = xmin + ijk[2] * dx
        xyz_surface = jnp.column_stack((xp, yp, zp))

        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(extra_points[:,0], extra_points[:,1], extra_points[:,2])
        # plt.savefig('octree.png')

        return xyz_surface

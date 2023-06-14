"""
======================= START OF LICENSE NOTICE =======================
  Copyright (C) 2022 Pouria Mistani and Samira Pakravan. All Rights Reserved

  NO WARRANTY. THE PRODUCT IS PROVIDED BY DEVELOPER "AS IS" AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DEVELOPER BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE PRODUCT, EVEN
  IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
======================== END OF LICENSE NOTICE ========================
  Primary Author: mistani

"""

import matplotlib

matplotlib.use("Agg")
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as onp


def plot_loss_epochs(epoch_store, loss_epochs, address, base_level, alt_res=True, name="solver_loss"):
    dir_name = os.path.dirname(address)
    if os.path.isdir(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    epoch_store = onp.array(epoch_store)
    loss_epochs = onp.array(loss_epochs)

    fig, ax = plt.subplots(figsize=(8, 8))

    # plt.plot(epoch_store[epoch_store%switching_interval - 1 ==0], loss_epochs[epoch_store%switching_interval - 1 ==0], color='k', label='whole domain')
    # plt.plot(epoch_store[epoch_store%switching_interval - 1 <0], loss_epochs[epoch_store%switching_interval - 1 <0], color='b', label='negative domain')
    # plt.plot(epoch_store[epoch_store%switching_interval - 1 >0], loss_epochs[epoch_store%switching_interval - 1 >0], color='r', label='positive domain')

    # plt.plot(epoch_store[epoch_store%switching_interval ==0], loss_epochs[epoch_store%switching_interval ==0], color='k', label='whole domain')
    # plt.plot(epoch_store[-1*( epoch_store%switching_interval) <0], loss_epochs[-1*(epoch_store%switching_interval) <0], color='b', label='negative domain')

    if alt_res:
        ax.plot(
            epoch_store[epoch_store % 4 == 0],
            loss_epochs[epoch_store % 4 == 0],
            color="k",
            label=r"$\rm level=\ $" + str(base_level),
        )
        ax.plot(
            epoch_store[epoch_store % 4 == 1],
            loss_epochs[epoch_store % 4 == 1],
            color="b",
            label=r"$\rm level=\ $" + str(base_level + 1),
        )
        ax.plot(
            epoch_store[epoch_store % 4 == 2],
            loss_epochs[epoch_store % 4 == 2],
            color="r",
            label=r"$\rm level=\ $" + str(base_level + 2),
        )
        ax.plot(
            epoch_store[epoch_store % 4 == 3],
            loss_epochs[epoch_store % 4 == 3],
            color="g",
            label=r"$\rm level=\ $" + str(base_level + 3),
        )

    else:
        ax.plot(
            epoch_store,
            loss_epochs,
            color="k",
            label=r"$\rm level\ =\ $" + str(base_level),
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\rm epoch$", fontsize=20)
    ax.set_ylabel(r"$\rm loss$", fontsize=20)
    plt.legend(fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=20)
    plt.tight_layout()
    filename = os.path.join(address, name + ".png")
    plt.savefig(filename)
    plt.close()


# alpha is transparency
def get_rgba(V, alpha=1):
    Vx = V[:, 0]
    Vy = V[:, 1]
    Vz = V[:, 2]
    colors = []
    for vx, vy, vz in zip(Vx, Vy, Vz):
        norm = (vx**2 + vy**2 + vz**2) ** 0.5
        col = (onp.array([vx / norm, vy / norm, vz / norm, alpha]) * 255).astype(onp.uint8)
        colors.append(col)
    return colors


def animate(log, scale_fac=0.05):
    Rs = log["R"]
    vels = log["V"]
    rgba = get_rgba(vels[0])

    times = log["t"]
    X = Rs[0][:, 0]
    Y = Rs[0][:, 1]
    Z = Rs[0][:, 2]
    fig = mlab.gcf()

    p = mlab.points3d(X, Y, Z, scale_factor=scale_fac)

    @mlab.animate(delay=100)
    def anim():
        f = mlab.gcf()
        while True:
            for i in range(1, len(times)):
                rgba = get_rgba(vels[i])
                p.mlab_source.set(
                    x=onp.array(Rs[i][:, 0]),
                    y=onp.array(Rs[i][:, 1]),
                    z=onp.array(Rs[i][:, 2]),
                )
                yield

    anim()
    mlab.show()


# --- FIELD DATA


def plot3D_field(gstate, U):
    X, Y, Z = onp.meshgrid(gstate.x, gstate.y, gstate.z, indexing="ij")
    # mlab.points3d(X.flatten(), Y.flatten(), Z.flatten(), U, colormap="Spectral")
    ff = onp.array(U.reshape(X.shape))
    mlab.contour3d(ff, contours=40, transparent=True)
    mlab.show()


def animate_field(gstate, log, **kwargs):
    X, Y, Z = onp.meshgrid(gstate.x, gstate.y, gstate.z, indexing="ij")
    U = log["U"][0]
    times = log["t"]

    # fig = mlab.gcf()
    fig = mlab.figure(size=(800, 800), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    ff = onp.array(U.reshape(X.shape))
    # p = mlab.points3d(X.flatten(), Y.flatten(), Z.flatten(), U, colormap="Spectral", scale_factor=scale_fac, vmin=-3.0, vmax=3.)
    p = mlab.contour3d(ff, **kwargs)

    # def make_frame(t):
    #     ff = onp.array(log['U'][t].reshape(X.shape))
    #     p.mlab_source.set(scalars=ff)
    #     return mlab.screenshot(antialiased=True)
    # duration = 5
    # animation = mpy.VideoClip(make_frame, duration=duration).resize(0.5)
    # animation.write_gif("Rotated_sym_distri_static_Python.gif", fps=20)
    # mlab.close(fig)

    @mlab.animate(delay=100)
    def anim():
        f = mlab.gcf()
        while True:
            for i in range(1, len(times)):
                ff = onp.array(log["U"][i].reshape(X.shape))
                p.mlab_source.set(scalars=ff)
                yield

    anim()
    mlab.show()


def plot_slice(gstate, log, time_indx=0, z_indx=-1):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contour(onp.array(log["U"][time_indx].reshape(gstate.shape())[:, :, z_indx]))
    plt.show()


def plot_slice_animation(gstate, log, **kwargs):
    X, Y, Z = onp.meshgrid(gstate.x, gstate.y, gstate.z, indexing="ij")
    # X = onp.swapaxes(X, 0, 1)
    # Y = onp.swapaxes(Y, 0, 1)
    # Z = onp.swapaxes(Z, 0, 1)
    U_block = onp.array(log["U"].reshape(((-1,) + gstate.shape())))
    zidx = len(gstate.z) // 2

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(i):
        ax.clear()
        ax.contour(X[:, :, zidx], Y[:, :, zidx], U_block[i, :, :, zidx], **kwargs)
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("y", fontsize=20)

    ani = animation.FuncAnimation(fig, update, frames=10, interval=500)
    ani.save("advection.gif", writer="pillow")

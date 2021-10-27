from mayavi import mlab
import numpy as onp

def plot(R):
    X = R[:,0]; Y=R[:,1]; Z=R[:,2]
    p = mlab.points3d(X, Y, Z)
    mlab.show()


def animate(log):
    Rs = log['R']
    vels = log['V']
    times = log['t']
    X = Rs[0][:,0]; Y=Rs[0][:,1]; Z=Rs[0][:,2]
    fig = mlab.gcf()
    p = mlab.points3d(X, Y, Z)
    @mlab.animate(delay=100)
    def anim():
        f = mlab.gcf()
        while True:
            for i in range(1, len(times)):
                p.mlab_source.set(x=onp.array(Rs[i][:,0]), y=onp.array(Rs[i][:,1]), z=onp.array(Rs[i][:,2]))
                yield
    anim()
    mlab.show()
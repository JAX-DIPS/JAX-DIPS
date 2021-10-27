from mayavi import mlab
import numpy as onp

def plot(R):
    X = R[:,0]; Y=R[:,1]; Z=R[:,2]
    p = mlab.points3d(X, Y, Z)
    mlab.show()

# alpha is transparency
def get_rgba(V, alpha=1):
    Vx = V[:,0]; Vy = V[:,1]; Vz = V[:,2]
    colors = []
    for vx, vy, vz in zip(Vx, Vy, Vz):
        norm = (vx**2 + vy**2 + vz**2)**0.5
        col = (onp.array([vx/norm, vy/norm, vz/norm, alpha])*255).astype(onp.uint8)
        colors.append(col)
    return colors


def animate(log, scale_fac=0.05):
    Rs = log['R']
    vels = log['V']
    rgba = get_rgba(vels[0])

    times = log['t']
    X = Rs[0][:,0]; Y=Rs[0][:,1]; Z=Rs[0][:,2]
    fig = mlab.gcf()
    
    p = mlab.points3d(X, Y, Z, scale_factor=scale_fac)

    @mlab.animate(delay=100)
    def anim():
        f = mlab.gcf()
        while True:
            for i in range(1, len(times)):
                rgba = get_rgba(vels[i])
                p.mlab_source.set(x=onp.array(Rs[i][:,0]), y=onp.array(Rs[i][:,1]), z=onp.array(Rs[i][:,2]))
                yield
    anim()
    mlab.show()
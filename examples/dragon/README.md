The SDF was downloaded from https://github.com/dcanelhas/sdf-dragon


# sdf-dragon
A signed distance field embedding of the well-known dragon 3D model, courtesy of [Stanford University Computer Graphics Laboratory](https://graphics.stanford.edu/data/3Dscanrep/)

<<<<<<< HEAD
The original model contains some small holes. To create a manifold (water-tight) mesh, a Poisson reconstruction was first performed, using [Meshlab](www.meshlab.net), with a depth of 10. 
=======
The original model contains some small holes. To create a manifold (water-tight) mesh, a Poisson reconstruction was first performed, using [Meshlab](www.meshlab.net), with a depth of 10.
>>>>>>> release

The final signed distance field embedding was computed using [SDFGen](https://github.com/christopherbatty/SDFGen), with the following parameters (0.001 units per voxel and 16 voxels of padding around the model bounds):

```./SDFGen dragon.obj 0.001 16```

The resulting SDF available through this repository, as a VTI file (visualization toolkit image). It can be opened using [ParaView](http://www.paraview.org/).

[![SDF Embedding, in ParaView](https://i.ytimg.com/vi/LGhUYjX-Ly0/0.jpg)](https://youtu.be/LGhUYjX-Ly0 "SDF Embedding, in ParaView")



# Loading the CSV file:
The grid can be loaded by

```
    xmin = 0.118; xmax = 0.353
    ymin = 0.088; ymax = 0.263
    zmin = 0.0615; zmax = 0.1835
    Nx = 236; Ny = 176; Nz = 123
<<<<<<< HEAD
    
=======

>>>>>>> release
    xc = jnp.linspace(xmin, xmax, Nx, dtype=f32)
    yc = jnp.linspace(ymin, ymax, Ny, dtype=f32)
    zc = jnp.linspace(zmin, zmax, Nz, dtype=f32)
    dx = xc[1] - xc[0]
    gstate = init_mesh_fn(xc, yc, zc)
    R = gstate.R

    dragon_host = onp.loadtxt(currDir + '/dragonian.csv')
    dragon = jnp.array(dragon_host)
<<<<<<< HEAD
```
=======
```
>>>>>>> release

"""
See https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
"""

from evtk.hl import rectilinearToVTK, imageToVTK, structuredToVTK
import numpy as onp
import pdb


def write_vtk(gstate, U):

    # imageToVTK("./image", cellData = {"pressure" : pressure}, pointData = {"temp" : temp} )
    X, Y, Z = onp.meshgrid(gstate.x, gstate.y, gstate.z, indexing='ij')
    
    # imageToVTK("./solution", pointData = {"sol" : } )
    for i, sol in enumerate(U):
        ff = onp.array(sol.reshape(X.shape))
        structuredToVTK('results/solution'+str(i).zfill(4), X, Y, Z, pointData={"sol" : ff})
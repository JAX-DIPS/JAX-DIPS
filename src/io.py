"""
See https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
"""

import os
from dataclasses import field
from evtk.hl import rectilinearToVTK, imageToVTK, structuredToVTK
import numpy as onp
import pdb


def write_vtk(gstate, log, maxsteps=None):

    # imageToVTK("./image", cellData = {"pressure" : pressure}, pointData = {"temp" : temp} )
    X, Y, Z = onp.meshgrid(gstate.x, gstate.y, gstate.z, indexing='ij')

    # imageToVTK("./solution", pointData = {"sol" : } )
    num_steps = len(log['U'])
    for i in range(num_steps):
        print(f'writing timestep {i}')
        sol = log['U'][i]
        vx = log['V'][i,:,0]
        vy = log['V'][i,:,1]
        vz = log['V'][i,:,2]

        ff = onp.array(sol.reshape(X.shape))
        vx = onp.array(vx.reshape(X.shape))
        vy = onp.array(vy.reshape(X.shape))
        vz = onp.array(vz.reshape(X.shape))

        structuredToVTK('results/solution'+str(i).zfill(4), X, Y, Z, pointData={"sol" : ff, "vx" : vx, "vy" : vy, "vz" : vz})
        if maxsteps:
            if i >= maxsteps-1:
                break


def write_vtk_solution(gstate, log, address = 'results/', maxsteps=None):

    # imageToVTK("./image", cellData = {"pressure" : pressure}, pointData = {"temp" : temp} )
    X, Y, Z = onp.meshgrid(gstate.x, gstate.y, gstate.z, indexing='ij')

    # imageToVTK("./solution", pointData = {"sol" : } )
    num_steps = len(log['U'])
    for i in range(num_steps):
        print(f'writing timestep {i}')
        sol = log['U'][i]

        ff = onp.array(sol.reshape(X.shape))

        structuredToVTK(address + '/solution'+str(i).zfill(4), X, Y, Z, pointData={"sol" : ff})
        if maxsteps:
            if i >= maxsteps-1:
                break


def write_vtk_manual(gstate, field_dict, filename='results/manual_dump'):
    # field_dict = {name : value, ...}
    X, Y, Z = onp.meshgrid(gstate.x, gstate.y, gstate.z, indexing='ij')
    host_dict = {}
    for field_name in field_dict.keys():
        host_dict[field_name] = onp.array(field_dict[field_name])
    structuredToVTK(filename, X, Y, Z, pointData=host_dict)


def write_vtk_log(gstate, log, address = 'results/', maxsteps=None):

    X, Y, Z = onp.meshgrid(gstate.x, gstate.y, gstate.z, indexing='ij')

    num_steps = len(log)
    keys = log.keys()
    keys.remove('t')
    host_dict = {}

    os.makedirs(address, exist_ok=True)
    for i in range(num_steps):
        print(f'writing timestep {i}')

        for key in keys:
            ff = onp.array(log.get(key, i).reshape(X.shape))
            host_dict[key] = ff

        structuredToVTK(address + '/solution'+str(i).zfill(4), X, Y, Z, pointData=host_dict)
        if maxsteps:
            if i >= maxsteps-1:
                break
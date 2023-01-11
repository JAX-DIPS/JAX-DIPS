import gdown
import os

gdown.download(id='1ylM3M40leD3hYmg4g65oWbB_dtge88NX', output = 'lucy.ply')
os.system("meshio convert lucy.ply lucy.obj")



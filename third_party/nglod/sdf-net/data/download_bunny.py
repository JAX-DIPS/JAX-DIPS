import gdown
import os

gdown.download(id='1x3-ehBRLr-qAaRdMnNjq8wWfC8oYFii0', output = 'bunny.ply')
os.system("meshio convert bunny.ply bunny.obj")
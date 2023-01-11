import gdown
import os

gdown.download(id='1eqAkhT_b61K45xDUXuQNGdKEwUriv6WO', output = 'Armadillo.ply')
os.system("meshio convert Armadillo.ply Armadillo.obj")
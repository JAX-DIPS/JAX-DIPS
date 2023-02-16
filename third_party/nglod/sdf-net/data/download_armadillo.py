import gdown
import os

# gdown.download(id='1eqAkhT_b61K45xDUXuQNGdKEwUriv6WO', output = 'Armadillo.ply')
# os.system("meshio convert Armadillo.ply Armadillo.obj")
os.system(
    "wget https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj"
)

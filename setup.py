import pathlib
import setuptools


_here = pathlib.Path(__file__).resolve().parent

name = "jax-dips"

version = "0.0.2"

author = "Pouria Mistani & Samira Pakravan"

author_email = "p.a.mistani@gmail.com"

description = "Differentiable 3D interfacial PDE solvers written in JAX using the Neural Bootstrapping Method."

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = "https://github.com/JAX-DIPS/JAX-DIPS"

license = "GNU LESSER GENERAL PUBLIC LICENSE v2.1"

classifiers = [
    "Development Status :: 1 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Free Software Foundation",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]

python_requires = "~=3.8"

install_requires = ["jax>=0.4.12", "dm-haiku>=0.0.9", "optax>=0.1.4", "pyevtk"]

setuptools.setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=url,
    license=license,
    classifiers=classifiers,
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    packages=setuptools.find_packages(),
)
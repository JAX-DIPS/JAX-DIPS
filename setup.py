import pathlib
import setuptools


_here = pathlib.Path(__file__).resolve().parent

name = "jax-dips"

version = "0.2.0"

author = "Pouria Mistani & Samira Pakravan"

author_email = "p.a.mistani@gmail.com"

description = "Differentiable 3D interfacial PDE solvers written in JAX using the Neural Bootstrapping Method."

with open(_here / "README.md", "r") as f:
    readme = f.read()

url = "https://github.com/JAX-DIPS/JAX-DIPS"

license = "GNU LESSER GENERAL PUBLIC LICENSE v2.1"

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
]

python_requires = "~=3.8"

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()
requirements.append("pyevtk")
requirements.pop(0)
requirements.pop(0)


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
    install_requires=requirements,
    packages=setuptools.find_packages(),
)

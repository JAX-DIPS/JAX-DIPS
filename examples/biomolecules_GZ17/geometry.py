from jax import numpy as jnp, jit
import jax
from examples.biomolecules_GZ17.units import *
import os

currDir = os.path.dirname(os.path.realpath(__file__))


def get_initial_level_set_fn(atom_xyz_rad_chg):
    @jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]

        def initialize(carry, xyzsc):
            phi_, = carry
            xc, yc, zc, sigma, _ = xyzsc
            phi_ = jnp.min(
                jnp.array(
                    [
                        phi_,
                        jnp.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2) - sigma,
                    ]
                )
            )
            phi_ = jnp.nan_to_num(phi_)
            return (phi_,), None

        phi_ = 1e9
        (phi_,), _ = jax.lax.scan(initialize, (phi_,), atom_xyz_rad_chg)

        return phi_

    return unperturbed_phi_fn

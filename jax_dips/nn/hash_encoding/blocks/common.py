import functools
from typing import Hashable, Sequence, Literal, Tuple, Any, get_args

import jax


CameraModelType = Literal[
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
]
PositionalEncodingType = Literal["identity", "frequency", "hashgrid"]
DirectionalEncodingType = Literal["identity", "sh"]
EncodingType = Literal[PositionalEncodingType, DirectionalEncodingType]
ActivationType = Literal[
    "exponential",
    "linear",
    "relu",
    "sigmoid",
    "thresholded_exponential",
    "truncated_exponential",
    "truncated_thresholded_exponential",
]

ColmapMatcherType = Literal["Exhaustive", "Sequential"]
LogLevel = Literal["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]
TransformsProvider = Literal["loaded", "orbit"]

DensityAndRGB = Tuple[jax.Array, jax.Array]
RGBColor = Tuple[float, float, float]
RGBColorU8 = Tuple[int, int, int]
FourFloats = Tuple[float, float, float, float]
Matrix4x4 = Tuple[FourFloats, FourFloats, FourFloats, FourFloats]


def empty_impl(clz):
    if "__dataclass_fields__" not in clz.__dict__:
        raise TypeError("class `{}` is not a dataclass".format(clz.__name__))

    fields = clz.__dict__["__dataclass_fields__"]

    def empty_fn(cls, /, **kwargs):
        """
        Create an empty instance of the given class, with untransformed fields set to given values.
        """
        for field_name, annotation in fields.items():
            if field_name not in kwargs:
                kwargs[field_name] = getattr(annotation.type, "empty", lambda: None)()
        return cls(**kwargs)

    setattr(clz, "empty", classmethod(empty_fn))
    return clz


def mkValueError(desc, value, type):
    variants = get_args(type)
    assert value not in variants
    return ValueError("Unexpected {}: '{}', expected one of [{}]".format(desc, value, "|".join(variants)))


def vmap_jaxfn_with(
    # kwargs copied from `jax.vmap` source
    in_axes: int | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: Hashable | None = None,
    axis_size: int | None = None,
    spmd_axis_name: Hashable | None = None,
):
    return functools.partial(
        jax.vmap,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
    )

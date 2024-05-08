import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxrenderer.geometry import Camera
from jaxrenderer.pipeline import render
from jaxrenderer.shaders.color import ColorExtraInput, ColorShader
from jaxrenderer.types import (
    Buffers,
)
from jaxrenderer.utils import transpose_for_display

eye = jnp.array((0.0, 0, 2))  # pyright: ignore[reportUnknownMemberType]
center = jnp.array((0.0, 0, 0))  # pyright: ignore[reportUnknownMemberType]
up = jnp.array((0.0, 1, 0))  # pyright: ignore[reportUnknownMemberType]

width = 1920
height = 1080
lowerbound = jnp.zeros(2, dtype=int)  # pyright: ignore[reportUnknownMemberType]
dimension = jnp.array((width, height))  # pyright: ignore[reportUnknownMemberType]
depth = jnp.array(255)  # pyright: ignore[reportUnknownMemberType]

camera: Camera = Camera.create(
    view=Camera.view_matrix(eye=eye, centre=center, up=up),
    projection=Camera.perspective_projection_matrix(
        fovy=90.0,
        aspect=1.0,
        z_near=-1.0,
        z_far=1.0,
    ),
    viewport=Camera.viewport_matrix(
        lowerbound=lowerbound,
        dimension=dimension,
        depth=depth,
    ),
)

buffers = Buffers(
    zbuffer=lax.full((width, height), 1.0),  # pyright: ignore[reportUnknownMemberType]
    targets=(
        lax.full((width, height, 3), 0.0),  # pyright: ignore[reportUnknownMemberType]
    ),
)
face_indices = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    (
        (0, 1, 2),
        (1, 3, 2),
        (0, 2, 4),
        (0, 4, 3),
        (2, 5, 1),
    )
)
position = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1, -1, 1.0),
        (-2, 0.0, 0.0),
    )
)
extra = ColorExtraInput(
    position=position,
    color=jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
        )
    ),
)

result = render(camera, ColorShader, buffers, face_indices, extra)


fig, axs = plt.subplots(  # pyright: ignore
    ncols=2,
    nrows=1,
    sharex=True,
    sharey=True,
    figsize=(16, 8),
)

axs[0].imshow(  # pyright: ignore[reportUnknownMemberType]
    transpose_for_display(result.zbuffer)
)
axs[1].imshow(  # pyright: ignore[reportUnknownMemberType]
    transpose_for_display(result.targets[0])
)
plt.show()

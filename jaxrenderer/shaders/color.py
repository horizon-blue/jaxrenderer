from functools import partial
from typing import NamedTuple, Tuple

from jaxtyping import Array, Float

from .._meta_utils import typed_jit as jit
from ..geometry import Camera, to_homogeneous
from ..shader import ID, PerVertex, Shader
from ..types import Colour as Color  # unnecessary type alias :P


class ColorExtraInput(NamedTuple):
    """Extra input for Color Shader.

    Attributes:
        - position: in world space, of each vertex.
        - color: of each vertex
    """

    position: Float[Array, "vertices 3"]  # in world space
    color: Float[Array, "vertices 3"]


class ColorExtraFragmentData(NamedTuple):
    """Color of each vertex on canvas"""

    color: Color


class ColorExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""

    canvas: Color


class ColorShader(
    Shader[ColorExtraInput, ColorExtraFragmentData, ColorExtraMixerOutput]
):
    @staticmethod
    @partial(jit, inline=True)
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: ColorExtraInput,
    ) -> Tuple[PerVertex, ColorExtraFragmentData]:
        position = to_homogeneous(extra.position[gl_VertexID])
        gl_Position = camera.to_clip(position)
        color = extra.color[gl_VertexID]

        return (PerVertex(gl_Position=gl_Position), ColorExtraFragmentData(color=color))

"""
This type stub file was generated by pyright.
"""

from collections.abc import Mapping

from matplotlib import _api

"""
Builtin colormaps, colormap handling utilities, and the `ScalarMappable` mixin.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :doc:`/tutorials/colors/colormap-manipulation` for examples of how to
  make colormaps.

  :doc:`/tutorials/colors/colormaps` an in-depth discussion of
  choosing colormaps.

  :doc:`/tutorials/colors/colormapnorms` for more details about data
  normalization.
"""
_LUTSIZE = ...

class ColormapRegistry(Mapping):
    r"""
    Container for colormaps that are known to Matplotlib by name.

    The universal registry instance is `matplotlib.colormaps`. There should be
    no need for users to instantiate `.ColormapRegistry` themselves.

    Read access uses a dict-like interface mapping names to `.Colormap`\s::

        import matplotlib as mpl
        cmap = mpl.colormaps['viridis']

    Returned `.Colormap`\s are copies, so that their modification does not
    change the global definition of the colormap.

    Additional colormaps can be added via `.ColormapRegistry.register`::

        mpl.colormaps.register(my_colormap)
    """
    def __init__(self, cmaps) -> None: ...
    def __getitem__(self, item): ...
    def __iter__(self): ...
    def __len__(self):  # -> int:
        ...
    def __str__(self) -> str: ...
    def __call__(self):  # -> list[Unknown]:
        """
        Return a list of the registered colormap names.

        This exists only for backward-compatibility in `.pyplot` which had a
        ``plt.colormaps()`` method. The recommended way to get this list is
        now ``list(colormaps)``.
        """
        ...
    def register(self, cmap, *, name=..., force=...):  # -> None:
        """
        Register a new colormap.

        The colormap name can then be used as a string argument to any ``cmap``
        parameter in Matplotlib. It is also available in ``pyplot.get_cmap``.

        The colormap registry stores a copy of the given colormap, so that
        future changes to the original colormap instance do not affect the
        registered colormap. Think of this as the registry taking a snapshot
        of the colormap at registration.

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap
            The colormap to register.

        name : str, optional
            The name for the colormap. If not given, ``cmap.name`` is used.

        force : bool, default: False
            If False, a ValueError is raised if trying to overwrite an already
            registered name. True supports overwriting registered colormaps
            other than the builtin colormaps.
        """
        ...
    def unregister(self, name):  # -> None:
        """
        Remove a colormap from the registry.

        You cannot remove built-in colormaps.

        If the named colormap is not registered, returns with no error, raises
        if you try to de-register a default colormap.

        .. warning::

            Colormap names are currently a shared namespace that may be used
            by multiple packages. Use `unregister` only if you know you
            have registered that name before. In particular, do not
            unregister just in case to clean the name before registering a
            new colormap.

        Parameters
        ----------
        name : str
            The name of the colormap to be removed.

        Raises
        ------
        ValueError
            If you try to remove a default built-in colormap.
        """
        ...
    def get_cmap(self, cmap):  # -> Colormap:
        """
        Return a color map specified through *cmap*.

        Parameters
        ----------
        cmap : str or `~matplotlib.colors.Colormap` or None

            - if a `.Colormap`, return it
            - if a string, look it up in ``mpl.colormaps``
            - if None, return the Colormap defined in :rc:`image.cmap`

        Returns
        -------
        Colormap
        """
        ...

_colormaps = ...

@_api.deprecated("3.7", alternative="``matplotlib.colormaps.register(name)``")
def register_cmap(name=..., cmap=..., *, override_builtin=...):  # -> None:
    """
    Add a colormap to the set recognized by :func:`get_cmap`.

    Register a new colormap to be accessed by name ::

        LinearSegmentedColormap('swirly', data, lut)
        register_cmap(cmap=swirly_cmap)

    Parameters
    ----------
    name : str, optional
       The name that can be used in :func:`get_cmap` or :rc:`image.cmap`

       If absent, the name will be the :attr:`~matplotlib.colors.Colormap.name`
       attribute of the *cmap*.

    cmap : matplotlib.colors.Colormap
       Despite being the second argument and having a default value, this
       is a required argument.

    override_builtin : bool

        Allow built-in colormaps to be overridden by a user-supplied
        colormap.

        Please do not use this unless you are sure you need it.
    """
    ...

get_cmap = ...

@_api.deprecated("3.7", alternative="``matplotlib.colormaps.unregister(name)``")
def unregister_cmap(name):  # -> None:
    """
    Remove a colormap recognized by :func:`get_cmap`.

    You may not remove built-in colormaps.

    If the named colormap is not registered, returns with no error, raises
    if you try to de-register a default colormap.

    .. warning::

      Colormap names are currently a shared namespace that may be used
      by multiple packages. Use `unregister_cmap` only if you know you
      have registered that name before. In particular, do not
      unregister just in case to clean the name before registering a
      new colormap.

    Parameters
    ----------
    name : str
        The name of the colormap to be un-registered

    Returns
    -------
    ColorMap or None
        If the colormap was registered, return it if not return `None`

    Raises
    ------
    ValueError
       If you try to de-register a default built-in colormap.
    """
    ...

class ScalarMappable:
    """
    A mixin class to map scalar data to RGBA.

    The ScalarMappable applies data normalization before returning RGBA colors
    from the given colormap.
    """

    def __init__(self, norm=..., cmap=...) -> None:
        """
        Parameters
        ----------
        norm : `.Normalize` (or subclass thereof) or str or None
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If a `str`, a `.Normalize` subclass is dynamically generated based
            on the scale with the corresponding name.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or `~matplotlib.colors.Colormap`
            The colormap used to map normalized data values to RGBA colors.
        """
        ...
    def to_rgba(self, x, alpha=..., bytes=..., norm=...):
        """
        Return a normalized rgba array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding `~numpy.ndarray` of rgba values will be returned,
        based on the norm and colormap set for this ScalarMappable.

        There is one special case, for handling images that are already
        rgb or rgba, such as might have been read from an image file.
        If *x* is an `~numpy.ndarray` with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an rgb or rgba array, and no mapping will be done.
        The array can be uint8, or it can be floating point with
        values in the 0-1 range; otherwise a ValueError will be raised.
        If it is a masked array, the mask will be ignored.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the preexisting alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the rgba
        array will be floats in the 0-1 range; if it is *True*,
        the returned rgba array will be uint8 in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
        ...
    def set_array(self, A):  # -> None:
        """
        Set the value array from array-like *A*.

        Parameters
        ----------
        A : array-like or None
            The values that are mapped to colors.

            The base class `.ScalarMappable` does not make any assumptions on
            the dimensionality and shape of the value array *A*.
        """
        ...
    def get_array(self):  # -> None:
        """
        Return the array of values, that are mapped to colors.

        The base class `.ScalarMappable` does not make any assumptions on
        the dimensionality and shape of the array.
        """
        ...
    def get_cmap(self):  # -> Colormap | None:
        """Return the `.Colormap` instance."""
        ...
    def get_clim(
        self,
    ):  # -> tuple[Unknown | float | Any | None, Unknown | float | Any | None]:
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        ...
    def set_clim(self, vmin=..., vmax=...):  # -> None:
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             The limits may also be passed as a tuple (*vmin*, *vmax*) as a
             single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        ...
    def get_alpha(self):  # -> float:
        """
        Returns
        -------
        float
            Always returns 1.
        """
        ...
    def set_cmap(self, cmap):  # -> None:
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        ...
    @property
    def norm(self):  # -> Normalize | Any | None:
        ...
    @norm.setter
    def norm(self, norm):  # -> None:
        ...
    def set_norm(self, norm):  # -> None:
        """
        Set the normalization instance.

        Parameters
        ----------
        norm : `.Normalize` or str or None

        Notes
        -----
        If there are any colorbars using the mappable for this norm, setting
        the norm of the mappable will reset the norm, locator, and formatters
        on the colorbar to default.
        """
        ...
    def autoscale(self):  # -> None:
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        ...
    def autoscale_None(self):  # -> None:
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        ...
    def changed(self):  # -> None:
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
        ...

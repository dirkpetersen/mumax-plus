import numpy as _np
import warnings as _w

import _mumaxpluscpp as _cpp
class VoronoiTessellator:

    def __init__(self, grainsize, seed=None, max_idx=256, region_of_center=None):
        """Create a Voronoi tessellator instance.
        
        This class is used to generate a Voronoi tessellation, which can
        be done using either the `generate` or the `coo_to_idx` method.

        **Important:** other methods in this class cannot be used unless
        `generate` has been called. E.g. retrieving a list of region
        indices requires a specified world and grid.

        Parameters
        ----------
        grainsize : float
            The average grain diameter.
        seed : int (default=None)
            The seed of the used random number generators. This seed affects
            the values of the generated region indices and the number and
            positions of the Voronoi centers.
        max_idx : int (default=256)
            The (inclusive) maximum region index within the tessellation. This value
            has no upper bound.

        region_of_center : callable, optional
            A function with signature tuple(float)->int which assigns region indices to
            generated Voronoi centers. If not specified, a random integer will be generated.

        """
        self.seed = seed if seed else _np.random.randint(1234567)
        self._impl = _cpp.VoronoiTessellator(grainsize, self.seed, max_idx, region_of_center)

    def generate(self, world, grid):
        """Generates a Voronoi tessellation.

        Returns an ndarray of shape (nz, ny, nx) which is filled
        with region indices."""
        self.tessellation = self._impl.generate(grid._impl, world.cellsize)

        return self.tessellation
    
    def coo_to_idx(self, x, y, z):
        """Returns the region index (int) of the given coordinate within the
        Voronoi tessellation."""
        return self._impl.coo_to_idx((x,y,z))

    @property
    def indexDictionary(self):
        """Create a dictionary where each region (key) is linked
        to a list of grid coordinates (value)."""

        if not hasattr(self, 'tessellation'):
            _w.warn(
            "The tessellation has not been generated yet."
             " Call `generate(world, grid)` before accessing `indexDictionary`.", UserWarning
            )
            return None

        from collections import defaultdict
        tessellation = self.tessellation
        _, nz, ny, nx = tessellation.shape
        
        idxs = tessellation.flatten()
        coords = _np.array(_np.meshgrid(range(nx), range(ny), range(nz), indexing='ij')
                          ).reshape(3, -1).T
        
        idxDict = defaultdict(list)
        
        for idx, coord in zip(idxs, coords):
            idxDict[idx].append(tuple(coord))
            
        idxDict = dict(idxDict)
        return idxDict

    @property
    def indices(self):
        """Returns list of unique region indices."""

        if not hasattr(self, 'tessellation'):
            _w.warn(
            "The tessellation has not been generated yet."
             " Call `generate(world, grid)` before accessing `indices`.", UserWarning
            )
            return None
        return _np.unique(_np.ravel(self.tessellation)).astype(int)

    @property
    def number_of_regions(self):
        """Returns number of unique region indices."""
        if not hasattr(self, 'tessellation'):
            _w.warn(
            "The tessellation has not been generated yet."
             " Call `generate(world, grid)` before accessing `number_of_regions`.", UserWarning
            )
            return None
        return _np.unique(_np.ravel(self.tessellation)).size

    # TODO: implement (C++) function which returns neighbouring regions of a certain idx

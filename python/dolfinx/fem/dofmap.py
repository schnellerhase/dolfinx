# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import numpy.typing as npt

from dolfinx import cpp as _cpp
from dolfinx.common import IndexMap
from dolfinx.fem import ElementDofLayout


class DofMap:
    """Degree-of-freedom map

    This class handles the mapping of degrees of freedom. It builds
    a dof map based on a FiniteElement on a specific mesh.
    """

    _cpp_object: _cpp.fem.DofMap

    def __init__(self, dofmap: _cpp.fem.DofMap):
        self._cpp_object = dofmap

    def cell_dofs(self, cell_index: int) -> npt.NDArray[np.int32]:
        """Cell local-global dof map

        Args:
            cell: The cell index.

        Returns:
            Local-global dof map for the cell (using process-local indices).
        """
        return self._cpp_object.cell_dofs(cell_index)

    @property
    def bs(self) -> int:
        """Returns the block size of the dofmap"""
        return self._cpp_object.bs

    @property
    def dof_layout(self) -> ElementDofLayout:
        """Layout of dofs on an element."""
        return self._cpp_object.dof_layout

    @property
    def index_map(self) -> IndexMap:
        """Index map that described the parallel distribution of the dofmap."""
        return self._cpp_object.index_map

    @property
    def index_map_bs(self) -> int:
        """Block size of the index map."""
        return self._cpp_object.index_map_bs

    @property
    def list(self) -> npt.NDArray[np.int32]:
        """Adjacency list with dof indices for each cell."""
        return self._cpp_object.map()

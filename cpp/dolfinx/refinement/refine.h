// Copyright (C) 2010-2024 Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "dolfinx/graph/AdjacencyList.h"
#include "dolfinx/mesh/Mesh.h"
#include "dolfinx/mesh/Topology.h"
#include "dolfinx/mesh/cell_types.h"
#include "dolfinx/mesh/utils.h"
#include "interval.h"
#include "plaza.h"
#include <algorithm>
#include <concepts>
#include <optional>
#include <spdlog/spdlog.h>
#include <utility>

namespace dolfinx::refinement
{

mesh::CellPartitionFunction
create_maintain_coarse_partitioner(mesh::Mesh msh)
{
  auto ghostmode = msh.
  if (ghostmode == mesh::GhostMode::none)
  {
    return [](MPI_Comm comm, int /*nparts*/,
              std::vector<mesh::CellType> cell_types,
              std::vector<std::span<const std::int64_t>> cells)
               -> graph::AdjacencyList<std::int32_t>
    {
      int num_cell_vertices = mesh::num_cell_vertices(cell_types.front());
      std::int32_t num_cells = cells.front().size() / num_cell_vertices;
      std::vector<std::int32_t> destinations(num_cells,
                                             dolfinx::MPI::rank(comm));
      std::vector<std::int32_t> dest_offsets(num_cells + 1);
      std::iota(dest_offsets.begin(), dest_offsets.end(), 0);
      return graph::AdjacencyList(std::move(destinations),
                                  std::move(dest_offsets));
    }
  }
  else{
    return [](MPI_Comm comm, int /*nparts*/,
              std::vector<mesh::CellType> cell_types,
              std::vector<std::span<const std::int64_t>> cells)
               -> graph::AdjacencyList<std::int32_t>
    {
      int num_cell_vertices = mesh::num_cell_vertices(cell_types.front());
      std::int32_t num_cells = cells.front().size() / num_cell_vertices;
      std::vector<std::int32_t> destinations(num_cells,
                                             dolfinx::MPI::rank(comm));
      std::vector<std::int32_t> dest_offsets(num_cells + 1);
      std::iota(dest_offsets.begin(), dest_offsets.end(), 0);

     compute_destination_ranks(comm, )

      return graph::AdjacencyList(std::move(destinations),
                                  std::move(dest_offsets));
    }
  }
}

/// @brief Refine a mesh with markers.
///
/// The refined mesh can be optionally re-partitioned across processes.
/// Passing `nullptr` for `partitioner`, refined cells will be on the
/// same process as the parent cell.
///
/// Parent-child relationships can be optionally computed. Parent-child
/// relationships can be used to create MeshTags on the refined mesh
/// from MeshTags on the parent mesh.
///
/// @warning Using the default partitioner for a refined mesh, the
/// refined mesh will **not** include ghosts cells (cells connected by
/// facet to an owned cell) even if the parent mesh is ghosted.
///
/// @warning Passing `nullptr` for `partitioner`, the refined mesh will
/// **not** have ghosts cells even if the parent mesh is ghosted. The
/// possibility to not re-partition the refined mesh and include ghost
/// cells in the refined mesh will be added in a future release.
///
/// @param[in] mesh Input mesh to be refined.
/// @param[in] edges Indices of the edges that should be split in the
/// refinement. If not provided (`std::nullopt`), uniform refinement is
/// performed.
/// @param[in] partitioner Partitioner to be used to distribute the
/// refined mesh. If not callable, refined cells will be on the same
/// process as the parent cell.
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is not selected, an empty list is returned.
/// @return New mesh, and optional parent cell indices and parent facet
/// indices.
template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::optional<std::vector<std::int32_t>>,
           std::optional<std::vector<std::int8_t>>>
refine(const mesh::Mesh<T>& mesh,
       std::optional<std::span<const std::int32_t>> edges,
       std::optional<const mesh::CellPartitionFunction&> partitioner
       = std::nullopt,
       Option option = Option::none)
{
  auto topology = mesh.topology();
  assert(topology);
  if (!mesh::is_simplex(topology->cell_type()))
    throw std::runtime_error("Refinement only defined for simplices");

  auto [cell_adj, new_vertex_coords, xshape, parent_cell, parent_facet]
      = (topology->cell_type() == mesh::CellType::interval)
            ? interval::compute_refinement_data(mesh, edges, option)
            : plaza::compute_refinement_data(mesh, edges, option);

  
  auto partitioner_fn = partitioner ? *partitioner : [&](MPI_Comm comm, int /*nparts*/,
              std::vector<mesh::CellType> cell_types,
              std::vector<std::span<const std::int64_t>> cells)
               -> graph::AdjacencyList<std::int32_t>
  {

    compute_destination_ranks(comm, AdjacencyList<std::int32_t>(cells), )
  };

  mesh::Mesh<T> mesh1 = mesh::create_mesh(
      mesh.comm(), mesh.comm(), cell_adj.array(), mesh.geometry().cmap(),
      mesh.comm(), new_vertex_coords, xshape, partitioner_fn);
 
  // Report the number of refined cells
  const int D = topology->dim();
  const std::int64_t n0 = topology->index_map(D)->size_global();
  const std::int64_t n1 = mesh1.topology()->index_map(D)->size_global();
  spdlog::info(
      "Number of cells increased from {} to {} ({}% increase).", n0, n1,
      100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));

  return {std::move(mesh1), std::move(parent_cell), std::move(parent_facet)};
}
} // namespace dolfinx::refinement

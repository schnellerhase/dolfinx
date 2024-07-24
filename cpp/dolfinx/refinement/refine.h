// Copyright (C) 2010-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "plaza.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>

namespace dolfinx::refinement
{
/// @brief Create a locally refined mesh.
///
/// @param[in] mesh Mesh to create a new, refined mesh from.
/// @param[in] edges Indices of the edges that should be split during
/// refinement. mesh::compute_incident_entities can be used to compute
/// the edges that are incident to other entities, e.g. incident to
/// cells.
/// @param[in] redistribute If `true` refined mesh is re-partitioned
/// across MPI ranks.
/// @return Refined mesh.
template <std::floating_point T>
mesh::Mesh<T> refine(const mesh::Mesh<T>& mesh,
                     std::optional<std::span<const std::int32_t>> edges,
                     bool redistribute = true)
{
  auto topology = mesh.topology();
  assert(topology);
  if (topology->cell_type() != mesh::CellType::triangle
      and topology->cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Refinement only defined for simplices");
  }

  auto [refined_mesh, parent_cell, parent_facet]
      = plaza::refine(mesh, edges, redistribute, plaza::Option::none);

  // Report the number of refined cellse
  const int D = topology->dim();
  const std::int64_t n0 = topology->index_map(D)->size_global();
  const std::int64_t n1 = refined_mesh.topology()->index_map(D)->size_global();
  spdlog::info(
      "Number of cells increased from {} to {} ({}% increase).", n0, n1,
      100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));

  return refined_mesh;
}

} // namespace dolfinx::refinement

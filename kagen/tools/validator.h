#pragma once

#include "kagen/kagen.h"

#include <mpi.h>

namespace kagen {
bool ValidateVertexRanges(const Edgelist& edge_list, VertexRange vertex_range, CommInterface& comm);

bool ValidateGraph(
    Graph& graph, bool allow_self_loops, bool allow_directed_graphs, bool allow_multi_edges, CommInterface& comm);

bool ValidateGraphInplace(
    Graph& graph, bool allow_self_loops, bool allow_directed_graphs, bool allow_multi_edges, CommInterface& comm);
} // namespace kagen

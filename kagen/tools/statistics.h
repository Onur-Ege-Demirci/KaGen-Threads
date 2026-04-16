#pragma once

#include "kagen/kagen.h"


#include <vector>

namespace kagen {
SInt FindNumberOfGlobalNodes(VertexRange vertex_range, CommInterface& comm);

SInt FindNumberOfGlobalEdges(const Edgelist& edges, CommInterface& comm);

std::vector<SInt> GatherNumberOfEdges(const Edgelist& edges, CommInterface& comm);

SInt    ReduceSum(SInt value, CommInterface& comm);
SInt    ReduceMin(SInt value, CommInterface& comm);
LPFloat ReduceMean(SInt value, CommInterface& comm);
SInt    ReduceMax(SInt value, CommInterface& comm);
LPFloat ReduceSD(SInt value, CommInterface& comm);

struct DegreeStatistics {
    SInt    min;
    LPFloat mean;
    SInt    max;
};

DegreeStatistics ReduceDegreeStatistics(const Edgelist& edges, SInt global_num_nodes, CommInterface& comm);

std::vector<SInt> ComputeDegreeBins(const Edgelist& edges, VertexRange vertex_range, CommInterface& comm);

double ComputeEdgeLocality(const Edgelist& edges, VertexRange vertex_range, CommInterface& comm);

SInt ComputeNumberOfGhostNodes(const Edgelist& edges, VertexRange vertex_range, CommInterface& comm);

void PrintBasicStatistics(
    const XadjArray& xadj, const AdjncyArray& adjncy, VertexRange vertex_range, bool root, CommInterface& comm);

void PrintBasicStatistics(const Edgelist& edges, VertexRange vertex_range, bool root, CommInterface& comm);

void PrintAdvancedStatistics(Edgelist& edges, VertexRange vertex_range, bool root, CommInterface& comm);
} // namespace kagen

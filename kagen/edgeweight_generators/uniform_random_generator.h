#pragma once

#include "kagen/context.h"
#include "kagen/edgeweight_generators/edge_weight_generator.h"
#include "kagen/kagen.h"
#include "kagen/communicators/communicator.h"

namespace kagen {
/*!
 * Locally draws a weight for each edge (u,v) from a pseudorandom number generator.
 * Then edges are exchanges such that on each PE the weights for the edge pair (u,v) and (v,u) are known. Then a common
 * consensus is found which of the weights to use for the undirected edge \{u, v\}.
 */
class UniformRandomEdgeWeightGenerator : public EdgeWeightGenerator {
public:
    UniformRandomEdgeWeightGenerator(EdgeWeightConfig config, CommInterface comm, VertexRange vertex_range);

    void GenerateEdgeWeights(const XadjArray& xadj, const AdjncyArray& adjncy, EdgeWeights& weights) final;
    void GenerateEdgeWeights(const Edgelist& edgelist, EdgeWeights& weights) final;

private:
    const EdgeWeightConfig config_;
    CommInterface               comm_;
    VertexRange            vertex_range_;
};
} // namespace kagen

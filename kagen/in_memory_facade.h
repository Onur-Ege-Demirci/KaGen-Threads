#pragma once

#include "kagen/context.h"
#include "kagen/kagen.h"
#include "kagen/kagen_communicator.h"

namespace kagen {
void  GenerateInMemoryToDisk(PGeneratorConfig config, KAGEN_Comm comm);
Graph GenerateInMemory(const PGeneratorConfig& config, GraphRepresentation representation, KAGEN_Comm comm);
} // namespace kagen

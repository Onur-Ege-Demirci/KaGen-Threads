#pragma once

#include "kagen/context.h"
#include "kagen/kagen.h"
#include "kagen/Communicatorunicator.h"

namespace kagen {
void  GenerateInMemoryToDisk(PGeneratorConfig config, CommInterface comm);
Graph GenerateInMemory(const PGeneratorConfig& config, GraphRepresentation representation, CommInterface comm);
} // namespace kagen

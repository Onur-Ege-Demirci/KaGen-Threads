#include "kagen/in_memory_facade.h"

#include "kagen/context.h"
#include "kagen/definitions.h"
#include "kagen/factories.h"
#include "kagen/generators/generator.h"
#include "kagen/io.h"
#include "kagen/tools/statistics.h"
#include "kagen/tools/validator.h"
#include "kagen/kagen_communicator.h"



#include <cmath>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <functional>
namespace kagen {






void GenerateInMemoryToDisk(PGeneratorConfig config, KAGEN_Comm comm) {
    PEID size, rank;
    comm.initialize_size(&size, &rank);
    
    auto graph = GenerateInMemory(config, GraphRepresentation::EDGE_LIST, comm);

    const auto t_start_io = std::chrono::steady_clock::now();

    const std::string base_filename = config.output_graph.filename;
    for (const FileFormat& format: config.output_graph.formats) {
        const auto& factory = GetGraphFormatFactory(format);

        const std::string filename   = (config.output_graph.extension && !factory->DefaultExtensions().empty())
                                           ? base_filename + "." + factory->DefaultExtensions().front()
                                           : base_filename;
        config.output_graph.filename = filename;

        GraphInfo info(graph, comm);
        auto      writer = factory->CreateWriter(config.output_graph, graph, info, rank, size);
        if (writer != nullptr) {
            //TODO_O
            WriteGraph(*writer.get(), config.output_graph, rank == ROOT && !config.quiet, comm);
        } else if (!config.quiet && rank == ROOT) {
            std::cout << "Warning: invalid file format " << format << " for writing; skipping\n";
        }
    }

    const auto t_end_io = std::chrono::steady_clock::now();
    

    if (!config.quiet && rank == ROOT) {
        std::chrono::duration<double> elapsed = t_end_io - t_start_io;
        std::cout << "IO took " << std::fixed << std::setprecision(3) << elapsed.count() << " seconds"
                  << std::endl;
    }
}

Graph GenerateInMemory(const PGeneratorConfig& config_template, GraphRepresentation representation, KAGEN_Comm comm) {

    PEID size, rank;
    comm.initialize_size(&size, &rank);

    const bool output_error = rank == ROOT;
    const bool output_info  = rank == ROOT && !config_template.quiet;

    if (output_info && config_template.print_header) {
        PrintHeader(config_template);
    }

    auto             factory = CreateGeneratorFactory(config_template.generator);
    PGeneratorConfig config;
    try {
        config = factory->NormalizeParameters(config_template, rank, size, output_info);
    } catch (const kagen::ConfigurationError& ex) {
        if (output_error) {
            std::cerr << "Error: " << ex.what() << "\n";
        }
        comm.barrier();
        comm.abort(1);
    }

    if (output_info) {
        std::cout << "Generating graph ... " << std::flush;
    }

    const auto t_start_graphgen = comm.getTime();

    auto generator = factory->Create(config, rank, size);
    generator->Generate(representation);
    comm.barrier();

    if (output_info) {
        std::cout << "OK" << std::endl;
    }

    const SInt num_edges_before_finalize = generator->GetNumberOfEdges();
    if (output_info) {
        std::cout << "Finalizing graph ... " << std::flush;
    }
    if (!config.skip_postprocessing) {
        generator->Finalize(comm);
        comm.barrier();
    }
    if (output_info) {
        std::cout << "OK" << std::endl;
    }
    const SInt num_edges_after_finalize = generator->GetNumberOfEdges();

    if (output_info) {
        std::cout << "Generating weights ... " << std::flush;
    }
    generator->GenerateEdgeWeights(config.edge_weights, comm);
    generator->GenerateVertexWeights(config.vertex_weights, comm);
    if (output_info) {
        std::cout << "OK" << std::endl;
    }

    const auto t_end_graphgen = comm.time();

    if (!config.skip_postprocessing && !config.quiet) {
        SInt num_global_edges_before, num_global_edges_after;
        comm.Reduce(&num_edges_before_finalize, &num_global_edges_before, 1, typeid(SInt) ,KAGEN_OP.SUM, ROOT);
        comm.Reduce(&num_edges_after_finalize, &num_global_edges_after, 1, typeid(SInt) ,KAGEN_OP.SUM, ROOT);

        if (num_global_edges_before != num_global_edges_after && output_info) {
            std::cout << "The number of edges changed from " << num_global_edges_before << " to "
                      << num_global_edges_after << " during finalization (= by "
                      << std::abs(
                             static_cast<SSInt>(num_global_edges_after) - static_cast<SSInt>(num_global_edges_before))
                      << ")" << std::endl;
        }
    }
    //TODO_O oh god
    if (config.permute) {
        generator->PermuteVertices(config, comm);
    }

    auto graph = generator->Take();

    // Validation
    if (config.validate_simple_graph) {
        if (output_info) {
            std::cout << "Validating graph ... " << std::flush;
        }
        //TODO_O oh god 2 
        bool success = ValidateGraph(graph, config.self_loops, config.directed, false, comm);
        comm.AllReduce(inplace_t, &success, 1, KAGEN_OP.LOR)
        if (!success) {
            if (output_error) {
                std::cerr << "Error: graph validation failed\n";
            }
            comm.abort(1);
        } else if (output_info) {
            std::cout << "OK" << std::endl;
        }
    }

    //TODO_O
    // Statistics
    if (!config.quiet) {
        if (output_info) {
            std::cout << "Generation took " << std::fixed << std::setprecision(3) << t_end_graphgen - t_start_graphgen
                      << " seconds" << std::endl;
            std::cout << "-------------------------------------------------------------------------------" << std::endl;
        }

        if (representation == GraphRepresentation::EDGE_LIST) {
            if (config.statistics_level >= StatisticsLevel::BASIC) {
                PrintBasicStatistics(graph.edges, graph.vertex_range, rank == ROOT, comm);
            }
            if (config.statistics_level >= StatisticsLevel::ADVANCED) {
                PrintAdvancedStatistics(graph.edges, graph.vertex_range, rank == ROOT, comm);
            }
        } else { // CSR
            if (config.statistics_level >= StatisticsLevel::BASIC) {
                PrintBasicStatistics(graph.xadj, graph.adjncy, graph.vertex_range, rank == ROOT, comm);
            }
            if (config.statistics_level >= StatisticsLevel::ADVANCED) {
                std::cout << "Advanced statistics are not available when generating the graph in CSR representation"
                          << std::endl;
            }
        }
        if (output_info && config.statistics_level != StatisticsLevel::NONE) {
            std::cout << "-------------------------------------------------------------------------------" << std::endl;
        }
    }

    return graph;
}
} // namespace kagen

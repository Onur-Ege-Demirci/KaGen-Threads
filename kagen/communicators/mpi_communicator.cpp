#include "mpi_communicator.h"

#include <execinfo.h>
#include <mpi.h>
#include <unistd.h>

#include "communicator.h"
#include <functional>
#include <iostream>
#include <map>
#include <typeindex>

void print_stacktrace() {
    void*  array[50];
    int    size    = backtrace(array, 50);
    char** strings = backtrace_symbols(array, size);

    std::cerr << "Stack trace:\n";
    for (int i = 0; i < size; i++) {
        std::cerr << strings[i] << "\n";
    }

    free(strings);
}



MPI_Communicator::MPI_Communicator() {
    // TODO_O is this safe?
    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
}

MPI_Communicator::MPI_Communicator(MPI_Comm comm_) {
    comm = comm_;
        
}



MPI_Communicator::~MPI_Communicator() {
    // TODO_O this feels super sus
    print_stacktrace();
    std::cerr << "MPI_Communicator destructor called, finalizing MPI\n";
}

void MPI_Communicator::GetWorldRank(int* rank) const {
    MPI_Comm_rank(comm, rank);
}
void MPI_Communicator::GetWorldSize(int* size) const {
    MPI_Comm_size(comm, size);
}

void MPI_Communicator::barrier() const {
    MPI_Barrier(comm);
}
void MPI_Communicator::abort(int code) {
    MPI_Abort(comm, code);
}



//TODO_O see mpi_communicator.h: ask about typing here: can I call on these types using the normal machinery?
void MPI_Communicator::CommitType(std::type_index type, size_t size) {
    if (dynamic_types.contains(type) && dynamic_types[type] != MPI_DATATYPE_NULL) {
        return;
    }
    MPI_Datatype mpi_type;
    MPI_Type_contiguous(size, MPI_BYTE, &mpi_type);
    MPI_Type_commit(&mpi_type);
    table[type] = mpi_type;
}


//TODO_O see mpi_communicator.h: ask about typing here: can I call on these types using the normal machinery?
void MPI_Communicator::FreeType(std::type_index type) {
    MPI_Type_free(&dynamic_types[type]);
    dynamic_types.erase(type);
}

double MPI_Communicator::getTime() const {
    return MPI_Wtime();
}

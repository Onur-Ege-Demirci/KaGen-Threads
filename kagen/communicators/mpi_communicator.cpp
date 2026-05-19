#include "mpi_communicator.h"

#include <mpi.h>

#include "./../kagen.h"
#include "communicator.h"
#include "communicator_interface.h"
#include <functional>
#include <map>
#include <typeindex>
#include <execinfo.h>
#include <unistd.h>
#include <iostream>

void print_stacktrace() {
    void* array[50];
    int size = backtrace(array, 50);
    char** strings = backtrace_symbols(array, size);

    std::cerr << "Stack trace:\n";
    for (int i = 0; i < size; i++) {
        std::cerr << strings[i] << "\n";
    }

    free(strings);
}



const std::unordered_map<std::type_index, MPI_Datatype> MPI_Communicator::builtin_types = {
        {std::type_index(typeid(int)), MPI_INT},
        {std::type_index(typeid(double)), MPI_DOUBLE},
        {std::type_index(typeid(unsigned int)), MPI_UNSIGNED},
        {std::type_index(typeid(long long)), MPI_LONG_LONG},
        {std::type_index(typeid(unsigned long long)), MPI_UNSIGNED_LONG_LONG},
        {std::type_index(typeid(long double)), MPI_LONG_DOUBLE},
        {std::type_index(typeid(uint64_t)), MPI_UINT64_T},
        {std::type_index(typeid(int64_t)), MPI_INT64_T},
        {std::type_index(typeid(bool)), MPI_C_BOOL},
        {std::type_index(typeid(uint8_t)), MPI_BYTE}
    };



MPI_Datatype MPI_Communicator::getMPIType(const std::type_info& type) {
    return getMPIType(std::type_index(type));
    
}

MPI_Datatype MPI_Communicator::getMPIType(std::type_index type) {
    if (builtin_types.contains(type)) {
        return builtin_types.at(type);
        
    }
    else if(dynamic_types.contains(type)){
        return dynamic_types.at(type);
    }
    throw std::runtime_error("MPI_Communicator does not support type " + std::string(type.name()));
    print_stacktrace();
}

static MPI_Op getMPIOp(CommOp op) {
    switch (op) {
        case CommOp::LOR:
            return MPI_LOR;
            break;
        case CommOp::MAX:
            return MPI_MAX;
            break;
        case CommOp::MIN:
            return MPI_MIN;
            break;
        case CommOp::SUM:
            return MPI_SUM;
            break;
    }
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
    //TODO_O this feels super sus.
    for(auto& [key, value] : dynamic_types) {
        FreeType(key);
    }
    print_stacktrace();
    std::cerr << "MPI_Communicator destructor called, finalizing MPI\n";
}

void MPI_Communicator::GetWorldRank(int* rank) {
    MPI_Comm_rank(comm, rank);
}
void MPI_Communicator::GetWorldSize(int* size) {
    MPI_Comm_size(comm, size);
}

void MPI_Communicator::barrier() {
    MPI_Barrier(comm);
}
void MPI_Communicator::abort(int code) {
    MPI_Abort(comm, code);
}
void MPI_Communicator::Reduce(
    const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) {
    MPI_Reduce(sendbuf, recvbuf, count, getMPIType(type), getMPIOp(op), root, comm);
}
void MPI_Communicator::Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) {
    MPI_Reduce(MPI_IN_PLACE, recvbuf, static_cast<int>(count), getMPIType(type), getMPIOp(op), root, comm);
}
void MPI_Communicator::Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) {
    MPI_Allreduce(sendbuf, recvbuf, count, getMPIType(type), getMPIOp(op), comm);
}
void MPI_Communicator::Allreduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op) {
    MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, getMPIType(type), getMPIOp(op), comm);
}

void MPI_Communicator::Allgather(
    const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
    const std::type_info& recv_type) {
    MPI_Allgather(sendbuf, sendcount, getMPIType(send_type), recvbuf, recvcount, getMPIType(recv_type), comm);
}

void MPI_Communicator::Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type) {
    int flag;
    MPI_Finalized(&flag);
    if (flag) {
        print_stacktrace();
        std::cerr << "MPI_Finalize has already been called, cannot perform Allgather with MPI_IN_PLACE\n" << recv_type.name() << "\n";
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvbuf, recvcount, getMPIType(recv_type), comm);
}

void MPI_Communicator::AllgatherV(
    const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[],
    const int displs[], const std::type_info& recv_type) {
    MPI_Allgatherv(sendbuf, sendcount, getMPIType(send_type), recvbuf, recvcounts, displs, getMPIType(recv_type), comm);
}

void MPI_Communicator::Broadcast(void* buffer, int count, const std::type_info& type, int root) {
    MPI_Bcast(buffer, count, getMPIType(type), root, comm);
}
void MPI_Communicator::Alltoall(
    const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
    const std::type_info& recv_type) {
    MPI_Alltoall(sendbuf, sendcount, getMPIType(send_type), recvbuf, recvcount, getMPIType(recv_type), comm);
}
void MPI_Communicator::AlltoallV(
    const void* sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type, void* recvbuf,
    const int recvcounts[], const int rdispls[], const std::type_info& recv_type) {
    MPI_Alltoallv(
        sendbuf, sendcounts, sdispls, getMPIType(send_type), recvbuf, recvcounts, rdispls, getMPIType(recv_type), comm);
}

void MPI_Communicator::Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) {
    MPI_Exscan(sendbuf, recvbuf, count, getMPIType(type), getMPIOp(op), comm);
}
void MPI_Communicator::CommitType(std::type_index type, size_t size) {
    if (dynamic_types.contains(type) && dynamic_types[type] != MPI_DATATYPE_NULL) {
        return;
    } 
    MPI_Datatype mpi_type;
    MPI_Type_contiguous(size, MPI_BYTE, &mpi_type);
    MPI_Type_commit(&mpi_type);
    dynamic_types[type] = mpi_type;
}


void MPI_Communicator::FreeType(std::type_index type) {
    MPI_Type_free(&dynamic_types[type]);
    dynamic_types.erase(type);
}

double MPI_Communicator::getTime() {
    return MPI_Wtime();
}

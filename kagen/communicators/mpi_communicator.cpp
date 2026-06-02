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

template <typename T>
void Reduce(std::span<const T> sendbuf, std::span<T> recvbuf, CommOp op, int root) {
    
    dispatch([&](auto& c) { c.Reduce(sendbuf, recvbuf, op, root); });
}

template <typename T>
void Reduce(inplace_t, std::span<T> recvbuf, CommOp op, int root) {
    dispatch([&](auto& c) { c.Reduce(inplace, recvbuf, op, root); });
}

template <typename T>
void Allreduce(std::span<const T> sendbuf, std::span<T> recvbuf, CommOp op) {
    dispatch([&](auto& c) { c.Allreduce(sendbuf, recvbuf, op); });
}

template <typename T>
void Allreduce(inplace_t, std::span<T> recvbuf, CommOp op) {
    dispatch([&](auto& c) { c.Allreduce(inplace, recvbuf, op); });
}

template <typename T>
void Allgather(std::span<const T> sendbuf, std::span<T> recvbuf) {
    dispatch([&](auto& c) { c.Allgather(sendbuf, recvbuf); });
}
template <typename T>
void Allgather(inplace_t, std::span<T> recvbuf) {
    dispatch([&](auto& c) { c.Allgather(inplace, recvbuf); });
}

template <typename T>
void AllgatherV(
    std::span<const T> sendbuf, std::span<T> recvbuf, std::span<const int> recvcounts, std::span<const int> displs) {
    dispatch([&](auto& c) { c.AllgatherV(sendbuf, recvbuf, recvcounts, displs); });
}

template <typename T>
void Broadcast(std::span<T> buffer, int root) {
    dispatch([&](auto& c) { c.Broadcast(buffer, root); });
}

template <typename T>
void Alltoall(std::span<const T> sendbuf, std::span<T> recvbuf) {
    dispatch([&](auto& c) { c.Alltoall(sendbuf, recvbuf); });
}
template <typename T>
void AlltoallV(
    std::span<const T> sendbuf, std::span<const int> sendcounts, std::span<const int> sdispls, std::span<T> recvbuf,
    std::span<const int> recvcounts, std::span<const int> rdispls) {
    dispatch([&](auto& c) { c.AlltoallV(sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls); });
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
    table[type] = mpi_type;
}

void MPI_Communicator::FreeType(std::type_index type) {
    MPI_Type_free(&dynamic_types[type]);
    dynamic_types.erase(type);
}

double MPI_Communicator::getTime() const {
    return MPI_Wtime();
}

#pragma once
#include "communicator.h"
#include "mpi_communicator.h"
#include "thread_communicator.h"
#include <chrono>
#include <memory>
#include <variant>
class CommInterface {
private:
    int                                                               rank;
    CommType                                                          type;
    std::variant<const MPI_Communicator*, const Thread_Communicator*> comm;

    template <typename Fn>

    decltype(auto) dispatch(Fn&& fn) {
        return std::visit([&](auto& ptr) -> decltype(auto) { return std::forward<Fn>(fn)(*ptr); }, comm);
    }
    // TODO_O change to raw pointers and let caller handle lifetime-
public:
    CommInterface(int rank, const Thread_Communicator& comm);
    CommInterface(int rank, const MPI_Communicator& comm);
    void GetRank(int*);
    void GetSize(int*);
    void barrier();

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
        std::span<const T> sendbuf, std::span<T> recvbuf, std::span<const int> recvcounts,
        std::span<const int> displs) {
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

    void   GetWorldRank(int* rank);
    double getTime();

    // TODO_O
    void Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op);
    void CommitType(std::type_index type, size_t size);
    void FreeType(std::type_index type);
    void abort(int code);
};
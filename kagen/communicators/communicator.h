#pragma once
#include "kagen.h"
#include <functional>

//inplace_t is an empty class type used to indicate that reduce / allreduce should be performed inplace.  [For when that is actually supported TODO_O]
struct inplace_t {};
constexpr inplace_t inplace{}; 


enum class CommOp { 
    SUM,
    MIN,
    MAX,
    LOR
};

//TODO_O subclasses again?
class Communicator {
    public:
        virtual void GetWorldRank(int *rank) = 0;
        virtual void GetWorldSize(int *size) = 0;
        virtual void Barrier() = 0;
        virtual void Abort(int code) = 0;
        virtual void Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) = 0;
        virtual void Reduce(inplace_t,           void* recvbuf, int count, const std::type_info& type, CommOp op, int root) = 0;
        virtual void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) = 0;
        virtual void Allgather(const void* sendbuf,  int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root) = 0;
        virtual void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root) = 0;
        virtual void AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[], const int displs[], const std::type_info& recv_type) = 0;
        virtual void Broadcast(void* buffer, int count, const std::type_info& type, int root) = 0;
        virtual void Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type, void *recvbuf, int recvcount, const std::type_info& recv_type) = 0;
        virtual void AlltoallV(const void *sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type, void *recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type) = 0;
};




#pragma once
#include "kagen.h"
#include <functional>

//inplace_t is an empty class type used to indicate that reduce / allreduce should be performed inplace.  [For when that is actually supported TODO_O]
constexpr inplace_t inplace{}; 


enum class CommOp { 
    SUM,
    MIN,
    MAX,
    LOR
};

//TODO_O subclasses again?
class Communicator {
    private:
        PEID size;
        CommType type;

    public:
        virtual void GetWorldSize(int *size) = 0;

        //TODO_O figure this out 
        int Execute(std::function<void(CommInterface)> func) {
            int rank;
            GetWorldRank(&rank);
            CommInterface interface() = CommInterface(rank, this);
            return func(interface);
        }
        virtual void Barrier() = 0;
        virtual void Abort(int code) = 0;
        virtual void Reduce(const void* sendbuf, void* recvbuf, int count, std::type_info type, COMM_OP op, int root) = 0;
        virtual void Reduce(inplace_t,           void* recvbuf, int count, std::type_info type, COMM_OP op, int root) = 0;
        virtual void Allreduce(const void* sendbuf, void* recvbuf, int count, std::type_info type, COMM_OP op) = 0;
        virtual void Allgather(const void* sendbuf,  int sendcount, std::type_info send_type, void* recvbuf, int recvcount, std::type_info recv_type, COMM_OP op, int root) = 0;
        virtual void Allgather(inplace_T, void* recvbuf, int recvcount, std::type_info recv_type, COMM_OP op, int root) = 0;
        virtual void AllgatherV(const void* sendbuf, int sendcount, std::type_info send_type, void* recvbuf, const int recvcounts[], const int displs[], std::type_info recv_type) = 0;
        virtual void Broadcast(void* buffer, int count, std::type_info type, int root) = 0;
        virtual void Alltoall(const void* sendbuf, int sendcount, std::type_info send_type, void *recvbuf, int recvcount, std::type_info recv_type) = 0;
        virtual void AlltoallV(const void *sendbuf, const int sendcounts[], const int sdispls[], std::type_info send_type, void *recvbuf, const int recvcounts[], const int rdispls[], std::type_info recv_type) = 0;
};




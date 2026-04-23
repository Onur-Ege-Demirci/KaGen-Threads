#pragma once
#include "communicator.h"
#include <chrono>

class CommInterface {
    private:
        int rank;
        std::shared_ptr<Communicator> comm;
    public:
        
        CommInterface(int rank, std::shared_ptr<Communicator> comm);
        void GetRank(int*);
        void GetSize(int*);
        void barrier();
        void CommitType(std::type_index type, size_t size);
        void FreeType(std::type_index type);
        void abort(int code);
        void Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root);
        void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root);
        void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op);
        void Allreduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op);
        void Allgather(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount, const std::type_info& recv_type);
        void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type);
        void AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[], const int displs[], const std::type_info& recv_type);
        void Broadcast(void* buffer, int count, const std::type_info& type, int root);
        void Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type, void *recvbuf, int recvcount, const std::type_info& recv_type);
        void AlltoallV(const void *sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type, void *recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type);
        void GetWorldRank(int* rank);
        void Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op); 
        double getTime();
};
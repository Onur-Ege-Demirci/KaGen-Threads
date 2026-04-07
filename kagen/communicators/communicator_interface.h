#include "kagen.h"
#include "communicator.h"


class CommInterface {
    private:
        PEID rank;
        Communicator* comm;
    public:
       
        CommInterface(PEID rank, Communicator* comm);
        int GetRank(int*);
        int GetSize(int*);
        void Barrier();
        void Abort(int code);
        void Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root);
        void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root);
        void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op);
        void Allgather(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root);
        void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root);
        void AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[], const int displs[], const std::type_info& recv_type);
        void Broadcast(void* buffer, int count, const std::type_info& type, int root);
        void Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type, void *recvbuf, int recvcount, const std::type_info& recv_type);
        void AlltoallV(const void *sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type, void *recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type);
        void GetWorldRank(int* rank);
};
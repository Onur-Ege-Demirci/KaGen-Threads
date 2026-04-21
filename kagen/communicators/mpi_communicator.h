#include "kagen/communicators/communicator.h"

class MPI_Communicator : public Communicator {
    public:
    MPI_Communicator();
    ~MPI_Communicator();

    void GetWorldRank(int *rank) override;
    void GetWorldSize(int *size) override;

    void barrier() override;
    void abort(int code) override;

    void Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) override;
    void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) override;

    void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) override;
    void Allreduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op) override;

    void Allgather(const void* sendbuf, int sendcount, const std::type_info& send_type,
                   void* recvbuf, int recvcount, const std::type_info& recv_type) override;

    void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type) override;

    void AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type,
                    void* recvbuf, const int recvcounts[], const int displs[],
                    const std::type_info& recv_type) override;

    void Broadcast(void* buffer, int count, const std::type_info& type, int root) override;

    void Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type,
                  void* recvbuf, int recvcount, const std::type_info& recv_type) override;

    void AlltoallV(const void* sendbuf, const int sendcounts[], const int sdispls[],
                   const std::type_info& send_type,
                   void* recvbuf, const int recvcounts[], const int rdispls[],
                   const std::type_info& recv_type) override;
    void Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) override; 
    double getTime() override;
};
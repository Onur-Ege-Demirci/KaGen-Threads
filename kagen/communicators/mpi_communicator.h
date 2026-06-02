#include "kagen/communicators/communicator.h"

#include <mpi.h>
#include "communicator.h"
#include <functional>
#include <map>
#include <typeindex>

class MPI_Communicator {
private:
    MPI_Comm comm;

    MPI_Datatype getMPIType(const std::type_info& type);

    MPI_Datatype getMPIType(std::type_index type);

public:
    MPI_Communicator();

    MPI_Communicator(MPI_Comm comm_);

    ~MPI_Communicator();

    void GetWorldRank(int* rank) const;
    void GetWorldSize(int* size) const;

    void barrier() const;
    void abort(int code);

    void
    Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) const;
    void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) const;

    void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) const;
    void Allreduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op) const;

    void Allgather(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
        const std::type_info& recv_type) const;

    void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type) const;

    void AllgatherV(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[],
        const int displs[], const std::type_info& recv_type) const;

    void Broadcast(void* buffer, int count, const std::type_info& type, int root) const;

    void Alltoall(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
        const std::type_info& recv_type) const;

    void AlltoallV(
        const void* sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type,
        void* recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type) const;
    void Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) const;

    void CommitType(std::type_index type, size_t size) const;
    void FreeType(std::type_index type) const;

    double getTime() const;
};
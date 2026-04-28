#include "kagen/communicators/communicator.h"

#include <mpi.h>

#include "./../kagen.h"
#include "communicator.h"
#include "communicator_interface.h"
#include <functional>
#include <map>
#include <typeindex>

class MPI_Communicator : public Communicator {
private:
    MPI_Comm                                          comm;
    std::unordered_map<std::type_index, MPI_Datatype> table = {
        {std::type_index(typeid(int)), MPI_INT},
        {std::type_index(typeid(double)), MPI_DOUBLE},
        {std::type_index(typeid(unsigned int)), MPI_UNSIGNED},
        {std::type_index(typeid(long long)), MPI_LONG_LONG},
        {std::type_index(typeid(unsigned long long)), MPI_UNSIGNED_LONG_LONG},
        {std::type_index(typeid(long double)), MPI_LONG_DOUBLE}};

    MPI_Datatype getMPIType(const std::type_info& type);

    MPI_Datatype getMPIType(std::type_index type);

public:
    MPI_Communicator();
    

    MPI_Communicator(MPI_Comm comm_);


    ~MPI_Communicator();

    void GetWorldRank(int* rank) override;
    void GetWorldSize(int* size) override;

    void barrier() override;
    void abort(int code) override;

    void
    Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) override;
    void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) override;

    void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) override;
    void Allreduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op) override;

    void Allgather(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
        const std::type_info& recv_type) override;

    void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type) override;

    void AllgatherV(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[],
        const int displs[], const std::type_info& recv_type) override;

    void Broadcast(void* buffer, int count, const std::type_info& type, int root) override;

    void Alltoall(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
        const std::type_info& recv_type) override;

    void AlltoallV(
        const void* sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type,
        void* recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type) override;
    void Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) override;

    void CommitType(std::type_index type, size_t size) override;
    void FreeType(std::type_index type) override;

    double getTime() override;
};
#include "kagen/communicators/communicator.h"
#include <kamping/types/mpi_type_traits.hpp>
#include <kamping/types/scoped_datatype.hpp>
#include <mpi.h>

#include "communicator.h"
#include <functional>
#include <map>
#include <typeindex>
#include <span>

class MPI_Communicator {
private:
    MPI_Comm comm;

    MPI_Datatype getMPIType(const std::type_info& type);

    MPI_Datatype getMPIType(std::type_index type);

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

public:
    MPI_Communicator();

    MPI_Communicator(MPI_Comm comm_);

    ~MPI_Communicator();

    void GetWorldRank(int* rank) const;
    void GetWorldSize(int* size) const;

    void barrier() const;
    void abort(int code);
    //TODO_O replace with builtintypes 
    template <typename T>
    void Reduce(std::span<const T> sendbuf, std::span<T> recvbuf, CommOp op, int root) {
        MPI_Datatype mpi_type = kamping::types::mpi_type_traits<T>::data_type();
        MPI_Reduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), mpi_type, getMPIOp(op), root, comm);
    }

    template <typename T>
    void Reduce(inplace_t, std::span<T> recvbuf, CommOp op, int root) {
        MPI_Datatype mpi_type = kamping::types::mpi_type_traits<T>::data_type();
        MPI_Reduce(MPI_IN_PLACE, recvbuf.data(), recvbuf.size(), mpi_type, getMPIOp(op), root, comm);
    }

    template <typename T>
    void Allreduce(std::span<const T> sendbuf, std::span<T> recvbuf, CommOp op) {
        MPI_Datatype mpi_type = kamping::types::mpi_type_traits<T>::data_type();
        MPI_Allreduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), mpi_type, getMPIOp(op), comm);
    }

    template <typename T>
    void Allreduce(inplace_t, std::span<T> recvbuf, CommOp op) {
        MPI_Datatype mpi_type = kamping::types::mpi_type_traits<T>::data_type();
        MPI_Allreduce(MPI_IN_PLACE, recvbuf.data(), recvbuf.size(), mpi_type, getMPIOp(op), comm);
    }

    template <typename T>
    void Allgather(std::span<const T> sendbuf, std::span<T> recvbuf) {
        MPI_Allgather(
            sendbuf.data(), sendbuf.size(), kamping::types::mpi_type_traits<T>::data_type(), recvbuf.data(),
            recvbuf.size(), kamping::types::mpi_type_traits<T>::data_type(), comm);
    }
    template <typename T>
    void Allgather(inplace_t, std::span<T> recvbuf) {
        MPI_Datatype mpi_type = kamping::types::mpi_type_traits<T>::data_type();
        MPI_Allgather(
            MPI_IN_PLACE, recvbuf.data(), recvbuf.size(), mpi_type, recvbuf.data(), recvbuf.size(), mpi_type, comm);
    }

    template <typename T>
    void AllgatherV(
        std::span<const T> sendbuf, std::span<T> recvbuf, std::span<const int> recvcounts,
        std::span<const int> displs) {
        MPI_Allgatherv(
            sendbuf.data(), sendbuf.size(), kamping::types::mpi_type_traits<T>::data_type(), recvbuf.data(),
            recvcounts.data(), displs.data(), kamping::types::mpi_type_traits<T>::data_type(), comm);
    }

    template <typename T>
    void Broadcast(std::span<T> buffer, int root) {
        MPI_Bcast(buffer.data(), buffer.size(), kamping::types::mpi_type_traits<T>::data_type(), root, comm);
    }

    template <typename T>
    void Alltoall(std::span<const T> sendbuf, std::span<T> recvbuf) {
        MPI_Alltoall(
            sendbuf.data(), sendbuf.size(), kamping::types::mpi_type_traits<T>::data_type(), recvbuf.data(),
            recvbuf.size(), kamping::types::mpi_type_traits<T>::data_type(), comm);
    }
    template <typename T>
    void AlltoallV(
        std::span<const T> sendbuf, std::span<const int> sendcounts, std::span<const int> sdispls, std::span<T> recvbuf,
        std::span<const int> recvcounts, std::span<const int> rdispls) {
        MPI_Alltoallv(
            sendbuf.data(), sendcounts.data(), sdispls.data(), kamping::types::mpi_type_traits<T>::data_type(),
            recvbuf.data(), recvcounts.data(), rdispls.data(), kamping::types::mpi_type_traits<T>::data_type(), comm);
    }

    template <typename T>
    void Exscan(std::span<const T> sendbuf, std::span<T> recvbuf, CommOp op) {
        MPI_Datatype mpi_type = kamping::types::mpi_type_traits<T>::data_type();
        MPI_Exscan(sendbuf.data(), recvbuf.data(), sendbuf.size(), mpi_type, getMPIOp(op), comm);
    }

    //TODO_O ask about typing here: can I call on these types using the normal machinery?
    //It is required that the types are committed before use, what happens with duplicate commits? To avoid unnecessary recommits, I should probably store the committed types. But if I do that and transfer the data accordingly, then in the reduce / etc. methods I need to make it aware of the datatype since they're handled differently. Short if/else check before every op?
    void CommitType(kamping::types::ScopedDatatype type);
    
    double getTime() const;
};
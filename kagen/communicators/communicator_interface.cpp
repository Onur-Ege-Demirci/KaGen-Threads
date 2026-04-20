#include "communicator.h"
#include "communicator_interface.h"




CommInterface::CommInterface(int rank, Communicator* comm) {
    this->rank = rank;
    this->comm = comm;
}
void CommInterface::GetRank(int* rank) {
    *rank = this->rank;
}
void CommInterface::GetSize(int* size) {
    comm->GetWorldSize(size);
}

void CommInterface::barrier() {
    comm->barrier();
}

void CommInterface::abort(int code) {
    comm->abort(code);
}

void CommInterface::Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) {
    comm->Reduce(sendbuf, recvbuf, count, type, op, root);
}

void CommInterface::Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) {
    comm -> Reduce(inplace, recvbuf, count, type, op, root);
}

void CommInterface::Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) {
    comm->Allreduce(sendbuf, recvbuf, count, type, op);
}

void CommInterface::Allgather(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount, const std::type_info& recv_type) {
    comm->Allgather(sendbuf, sendcount, send_type, recvbuf, recvcount, recv_type);
}

void CommInterface::Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type) {
    comm -> Allgather(inplace, recvbuf, recvcount, recv_type);
}

void CommInterface::AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[], const int displs[], const std::type_info& recv_type) {
    comm -> AllgatherV(sendbuf, sendcount, send_type, recvbuf, recvcounts, displs, recv_type);        
}

void CommInterface::Broadcast(void* buffer, int count, const std::type_info& type, int root) {
    comm->Broadcast(buffer, count, type, root);

}

void CommInterface::Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type, void *recvbuf, int recvcount, const std::type_info& recv_type) {
    comm -> Alltoall(sendbuf, sendcount, send_type, recvbuf, recvcount, recv_type);
}

void CommInterface::AlltoallV(const void *sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type, void *recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type) {
    comm -> AlltoallV(sendbuf, sendcounts, sdispls, send_type, recvbuf, recvcounts, rdispls, recv_type);
}

double CommInterface::getTime() {
    return comm -> getTime();
}


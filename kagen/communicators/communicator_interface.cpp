#include "communicator.h"
#include "communicator_interface.h"
#include "mpi_communicator.h"
#include "thread_communicator.h"





CommInterface::CommInterface(int rank, MPI_Communicator& comm) {
    this->rank = rank;
    this->comm = &comm;
}
CommInterface::CommInterface(int rank, Thread_Communicator& comm) {
    this->rank = rank;
    this->comm = &comm;
}


void CommInterface::GetRank(int* rank) {
    *rank = this->rank;
}
void CommInterface::GetSize(int* size) {
    dispatch([&](auto& c) {
        c.GetWorldSize(size);
    });
   
}

void CommInterface::barrier() {
    dispatch([&](auto& c) {
        c.barrier();
    });
}

void CommInterface::abort(int code) {
    dispatch([&](auto& c)) {
        c.abort(code);
    };

}

double CommInterface::getTime() {
    return comm -> getTime();
}

void CommInterface::Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) {
    comm -> Exscan(sendbuf, recvbuf, count, type, op);
}

void CommInterface::CommitType(std::type_index type, size_t size) {
    comm -> CommitType(type, size);
}
void CommInterface::FreeType(std::type_index type) {
    comm -> FreeType(type);
}

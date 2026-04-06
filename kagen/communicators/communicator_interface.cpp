#include "kagen.h"
#include "communicator.h"

using namespace kagen;

class CommInterface {
    private:
        PEID rank;
        Communicator* comm;
    public:
        CommInterface(PEID rank, Communicator* comm) {
            this->rank = rank;
            this->comm = comm;
        }
        int getRank() {
            return rank;
        }
        int getSize() {
            int size;
            comm->GetWorldSize(&size);
            return size;
        }

        void Barrier() {
            comm->Barrier();
        }

        void Abort(int code) {
            comm->Abort(code);
        }

        void Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) {
            comm->Reduce(sendbuf, recvbuf, count, type, op, root);
        }

        void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) {
            comm->Reduce(inplace, recvbuf, count, type, op, root);
        }

        void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) {
            comm->Allreduce(sendbuf, recvbuf, count, type, op);
        }

        void Allgather(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root) {
            comm->Allgather(sendbuf, sendcount, send_type, recvbuf, recvcount, recv_type, op, root);
        }

        void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root) {
            comm->Allgather(inplace, recvbuf, recvcount, recv_type, op, root);
        }

        void AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[], const int displs[], const std::type_info& recv_type) {
            comm->AllgatherV(sendbuf, sendcount, send_type, recvbuf, recvcounts, displs, recv_type);
        }

        void Broadcast(void* buffer, int count, const std::type_info& type, int root) {
            comm->Broadcast(buffer, count, type, root);
        }

        void Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type, void *recvbuf, int recvcount, const std::type_info& recv_type) {
            comm->Alltoall(sendbuf, sendcount, send_type, recvbuf, recvcount, recv_type);
        }

        void AlltoallV(const void *sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type, void *recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type) {
            comm->AlltoallV(sendbuf, sendcounts, sdispls, send_type, recvbuf, recvcounts, rdispls, recv_type);
        }


};
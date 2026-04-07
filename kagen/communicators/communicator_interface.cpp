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
            comm->Reduce(ConstBufferRef(sendbuf, count, &type), BufferRef(recvbuf, count, &type), op, root);
        }

        void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) {
            comm->Reduce(inplace, BufferRef(recvbuf, count, &type), op, root);
        }

        void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) {
            comm->Allreduce(ConstBufferRef(sendbuf, count, &type), BufferRef(recvbuf, count, &type), op);
        }

        void Allgather(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root) {
            comm->Allgather(ConstBufferRef(sendbuf, sendcount, &send_type), BufferRef(recvbuf, recvcount, &recv_type), op, root);
        }

        void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root) {
            comm->Allgather(inplace, BufferRef(recvbuf, recvcount, &recv_type), op, root);
        }

        void AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[], const int displs[], const std::type_info& recv_type) {
            comm->AllgatherV(ConstBufferRef(sendbuf, sendcount, &send_type), BufferRef(recvbuf, 0, &recv_type), recvcounts, displs);
        }

        void Broadcast(void* buffer, int count, const std::type_info& type, int root) {
            comm->Broadcast(BufferRef(buffer, count, &type), root);
        }

        void Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type, void *recvbuf, int recvcount, const std::type_info& recv_type) {
            comm->Alltoall(ConstBufferRef(sendbuf, sendcount, &send_type), BufferRef(recvbuf, recvcount, &recv_type));
        }

        void AlltoallV(const void *sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type, void *recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type) {
            comm->AlltoallV(ConstBufferRef(sendbuf, 0, &send_type), sendcounts, sdispls, BufferRef(recvbuf, 0, &recv_type), recvcounts, rdispls);
        }


};
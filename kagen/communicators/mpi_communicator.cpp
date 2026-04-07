#include "kagen.h"
#include "communicator.h"
#include "communicator_interface.h"
#include <map>
#include <functional>
class MPI_Communicator : Communicator {
    private:
        MPI_Comm comm;
    
        static MPI_Datatype getMPIType(const std::type_info& type) {
            //TODO_O ka💥💥💥l import
            return MPI_BYTE;  // Placeholder
        }
        static getMPIOp(CommOp op) {
            switch(op){
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
        MPI_Communicator() {
            MPI_Init();
            comm = MPI_COMM_WORLD;
        }
        ~MPI_Communicator() {
            MPI_Finalize();
        }
        void GetWorldRank(int *rank) override {
            MPI_Comm_rank(comm, rank);
        }
        void GetWorldSize(int *size) override  {
            MPI_Comm_size(comm, size);
        }

        void Barrier() override {
            MPI_Barrier(comm);
        }
        void Abort(int code) override {
            MPI_Abort(comm, code);
        }
        void Reduce(ConstBufferRef sendbuf, BufferRef recvbuf, CommOp op, int root) override {
            MPI_Reduce(sendbuf.data, recvbuf.data, static_cast<int>(recvbuf.count), getMPIType(*recvbuf.type_info), getMPIOp(op), root, comm);
        }
        void Reduce(inplace_t, BufferRef recvbuf, CommOp op, int root) override {
            MPI_Reduce(MPI_IN_PLACE, recvbuf.data, static_cast<int>(recvbuf.count), getMPIType(*recvbuf.type_info), getMPIOp(op), root, comm);
        }
        void Allreduce(ConstBufferRef sendbuf, BufferRef recvbuf, CommOp op) override {
            MPI_Allreduce(sendbuf.data, recvbuf.data, static_cast<int>(recvbuf.count), getMPIType(*recvbuf.type_info), getMPIOp(op), comm);
        }
        void Allgather(ConstBufferRef sendbuf, BufferRef recvbuf, CommOp op, int root) override {
            MPI_Allgather(sendbuf.data, static_cast<int>(sendbuf.count), getMPIType(*sendbuf.type_info), recvbuf.data, static_cast<int>(recvbuf.count), getMPIType(*recvbuf.type_info), comm);
        }
        void Allgather(inplace_t, BufferRef recvbuf, CommOp op, int root) override {
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvbuf.data, static_cast<int>(recvbuf.count), getMPIType(*recvbuf.type_info), comm);
        }
        
        void AllgatherV(ConstBufferRef sendbuf, BufferRef recvbuf, const int recvcounts[], const int displs[]) override {
            MPI_Allgatherv(sendbuf.data, static_cast<int>(sendbuf.count), getMPIType(*sendbuf.type_info), recvbuf.data, recvcounts, displs, getMPIType(*recvbuf.type_info), comm);
        }
   
        void Broadcast(BufferRef buffer, int root) override {
            MPI_Bcast(buffer.data, static_cast<int>(buffer.count), getMPIType(*buffer.type_info), root, comm);
        }
        void Alltoall(ConstBufferRef sendbuf, BufferRef recvbuf) override {
            MPI_Alltoall(sendbuf.data, static_cast<int>(sendbuf.count), getMPIType(*sendbuf.type_info), recvbuf.data, static_cast<int>(recvbuf.count), getMPIType(*recvbuf.type_info), comm);
        }
        void AlltoallV(ConstBufferRef sendbuf, const int sendcounts[], const int sdispls[], BufferRef recvbuf, const int recvcounts[], const int rdispls[]) override {
            MPI_AlltoallV(sendbuf.data, sendcounts, sdispls, getMPIType(*sendbuf.type_info), recvbuf.data, recvcounts, rdispls, getMPIType(*recvbuf.type_info), comm);
        }
    }
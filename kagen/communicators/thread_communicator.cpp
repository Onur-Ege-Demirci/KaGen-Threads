#include "kagen.h"
#include "communicator.h"

#include <vector>
#include <thread>

using std::vector;
using std::thread;
//TODO_O flip so the threads are made by the environment
class Thread_Communicator : Communicator {
    private:
        vector<thread>& threads;

        static getOp(CommOp op) {
            switch(op){
                case CommOp::LOR:
                    //TODO_O
                    break;
                case CommOp::MAX:
                    //TODO_O
                    break;
                case CommOp::MIN:
                    //TODO_O
                    break;
                case CommOp::SUM:
                    //TODO_O
                    break;
            }

        }
        

    public:
        Thread_Communicator(vector<thread>& threads) {
            this.threads = threads;
        }

        ~Thread_Communicator() {
            //TODO_O
        }
        void GetWorldSize(int *size) override  {
            size = threads.size;
        }

        void GetWorldRank(int *rank) override {
            //TODO_O offload to comm tokmak
        }

        int Execute(std::function<void(CommInterface)> func) {
            int rank;
            GetWorldRank(&rank);
            CommInterface interface() = CommInterface(rank, this);
            return func(interface);
        }
        void Barrier() override {
            MPI_Barrier(comm);
        }
        void Abort(int code) override {
            MPI_Abort(comm, code);
        }
        void Reduce(const void* sendbuf, void* recvbuf, int count, std::type_info type, COMM_OP op, int root) override {
            MPI_Reduce(sendbuf, recvbuf, count, getMPIType(type), getMPIOp(op), root, comm);
        }
        void Reduce(inplace_t,           void* recvbuf, int count, std::type_info type, COMM_OP op, int root) override {
            MPI_Reduce(MPI_IN_PLACE, recvbuf, count, getMPIType(type), getMPIOp(op), root, comm);
        }
        void Allreduce(const void* sendbuf, void* recvbuf, int count, std::type_info type, COMM_OP op) override {
            MPI_Allreduce(sendbuf, recvbuf, count, getMPIType(type), getMPIOp(op), comm);
        }
        void Allgather(const void* sendbuf,  int sendcount, std::type_info send_type, void* recvbuf, int recvcount, std::type_info recv_type, COMM_OP op, int root) override {
            MPI_Allgather(sendbuf, sendcount, getMPIType(send_type), recvbuf, recvcount, getMPIType(recv_type), comm);
        }
        void Allgather(inplace_T, void* recvbuf, int recvcount, std::type_info recv_type, COMM_OP op, int root) override {
            MPI_Allgather(MPI_IN_PLACE, sendcount, getMPIType(send_type), recvbuf, recvcount, getMPIType(recv_type), comm);
        }
        
        void AllgatherV(const void* sendbuf, int sendcount, std::type_info send_type, void* recvbuf, const int recvcounts[], const int displs[], std::type_info recv_type) override {
            MPI_Allgatherv(sendbuf, sendcount, getMPIType(send_type), recvbuf, recvcounts, displs, getMPIType(recv_type), comm);
        }
   
        void Broadcast(void* buffer, int count, std::type_info type, int root) {
            MPI_Bcast(buffer, count, getMPIType(type), root, comm);
        }
        void Alltoall(const void* sendbuf, int sendcount, std::type_info send_type, void *recvbuf, int recvcount, std::type_info recv_type) override {
            MPI_Alltoall(sendbuf, sendcount, getMPItype(send_type), recvbuf, recvcount, getMPItype(recv_type), comm);

        }
        void AlltoallV(const void *sendbuf, const int sendcounts[], const int sdispls[], std::type_info send_type, void *recvbuf, const int recvcounts[], 
            const int rdispls[], std::type_info recv_type) override {
            MPI_AlltoallV(sendbuf, sendcounts, sdispls, getMPItype(send_type), recvbuf, recvcounts, rdispls, getMPItype(recv_type), comm);
        }

    }
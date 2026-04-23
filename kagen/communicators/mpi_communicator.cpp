#include "./../kagen.h"
#include "communicator.h"
#include "communicator_interface.h"
#include <map>
#include <functional>
#include <typeindex>
#include <mpi.h>
class MPI_Communicator : public Communicator {
    private:
        MPI_Comm comm;
        std::unordered_map<std::type_index, MPI_Datatype> table = {
            {std::type_index(typeid(int)), MPI_INT},
            {std::type_index(typeid(double)), MPI_DOUBLE},
            {std::type_index(typeid(unsigned int)), MPI_UNSIGNED},
            {std::type_index(typeid(long long)), MPI_LONG_LONG},
            {std::type_index(typeid(unsigned long long)), MPI_UNSIGNED_LONG_LONG},
            {std::type_index(typeid(long double)), MPI_LONG_DOUBLE} 
};
        MPI_Datatype getMPIType(const std::type_info& type) {
            return table.at(std::type_index(type));
        }

        MPI_Datatype getMPIType(std::type_index type) {
            return table.at(type);
        }

        static MPI_Op getMPIOp(CommOp op) {
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
            //TODO_O is this safe? 
            MPI_Init(NULL, NULL);
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

        void barrier() override {
            MPI_Barrier(comm);
        }
        void abort(int code) override {
            MPI_Abort(comm, code);
        }
        void Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) override {
            MPI_Reduce(sendbuf, recvbuf, count, getMPIType(type), getMPIOp(op), root, comm);
        }
        void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) override {
            MPI_Reduce(MPI_IN_PLACE, recvbuf, static_cast<int>(count), getMPIType(type), getMPIOp(op), root, comm);
        }
        void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) override {
            MPI_Allreduce(sendbuf, recvbuf, count, getMPIType(type), getMPIOp(op), comm);
        }
        void Allreduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op) override {
            MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, getMPIType(type), getMPIOp(op), comm);
        }
        
        void Allgather(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount, const std::type_info& recv_type) override {
            MPI_Allgather(sendbuf, sendcount, getMPIType(send_type), recvbuf, recvcount, getMPIType(recv_type), comm);
        }
        
        void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type) override {
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recvbuf, recvcount, getMPIType(recv_type), comm);
        }
        
        void AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[], const int displs[], const std::type_info& recv_type) override {
            MPI_Allgatherv(sendbuf, sendcount, getMPIType(send_type), recvbuf, recvcounts, displs, getMPIType(recv_type), comm);
        }
   
        void Broadcast(void* buffer, int count, const std::type_info& type, int root) override {
            MPI_Bcast(buffer, count, getMPIType(type), root, comm);
        }
        void Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount, const std::type_info& recv_type) override {
            MPI_Alltoall(sendbuf, sendcount, getMPIType(send_type), recvbuf, recvcount, getMPIType(recv_type), comm);
        }
        void AlltoallV(const void* sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type, void* recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type) override {
            MPI_Alltoallv(sendbuf, sendcounts, sdispls, getMPIType(send_type), recvbuf, recvcounts, rdispls, getMPIType(recv_type), comm);
        }

        void Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) override {
            MPI_Exscan(sendbuf, recvbuf, count, getMPIType(type), getMPIOp(op), comm);
        }
        void CommitType(std::type_index type, size_t size) override {
            MPI_Datatype mpi_type;
            MPI_Type_contiguous(size, MPI_BYTE, &mpi_type);
            MPI_Type_commit(&mpi_type);
            table[type] = mpi_type;
        }


        //TODO_O does this even work? 
        void FreeType(std::type_index type) override {
            MPI_Type_free(&table[type]);
            table.erase(type);
        }

        double getTime() override {
            return MPI_Wtime();
        }
};
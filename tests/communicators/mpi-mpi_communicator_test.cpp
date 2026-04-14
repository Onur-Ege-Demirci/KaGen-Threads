#include <gtest/gtest.h>
#include "kagen/communicators/mpi_communicator.h"
#include "kagen/communicators/thread_communicator.h"
#include <vector>
#include "mpi.h"
using namespace std;
Thread_Communicator thread_comm;

vector<vector<int>> reduce_test_data = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12},
    {13, 14, 15, 16}
}
int test_count = 4;
vector<void*> recv_buffers;
void threadTest(int rank, int size) {
    vector<void*> thread_recvbufs;
    thread_recvbufs.emplace_back(malloc(reduce_test_data[rank].size() * sizeof(int)));
    thread_comm.Reduce(reduce_test_data[rank].data(), thread_recvbufs[0], CommOp::SUM, 0);
    if (rank == 0) {
            int* int_recvbuf = static_cast<int*>(thread_recvbufs[0]);
            int* int_mpi_recvbuf = static_cast<int*>(recv_buffers[0]);
            for (size_t i = 0; i < reduce_test_data[rank].size(); i++) {
                EXPECT_EQ(int_recvbuf[i], int_mpi_recvbuf[i]);
            }
    }
        
    
    thread_comm.barrier();
    vector<int> inplace_receive_buffer(reduce_test_data[rank].size());
    thread_comm.Reduce(inplace, inplace_receive_buffer.data(), CommOp::SUM, 0);
    thread_comm.barrier();
    thread_comm.Allreduce(reduce_test_data[rank].data(), thread_recvbufs[1], CommOp::SUM);
    thread_comm.barrier();
    inplace_receive_buffer = vector<int>(reduce_test_data[rank].size());
    thread_comm.Allreduce(inplace, inplace_receive_buffer.data(), CommOp::SUM);
    thread_comm.barrier();
    

}

int main() {
    MPI_Init();
    int rank, size = 4;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    ::testing::InitGoogleTest();
    for ( int i = 0 ; i< test_count; i++) {
        int* recvbuf = static_cast<int*>(malloc(reduce_test_data[rank].size() * sizeof(int)));
        recv_buffers.emplace_back(recvbuf);
    }
    MPI_Reduce(reduce_test_data[rank].data(), recv_buffers[0], reduce_test_data[rank].size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(reduce_test_data[rank].data(), recv_buffers[1], reduce_test_data[rank].size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Finalize();
   
    vector <thread> threads(size);
    for (int t = 0; t < size; t++) {
        threads[t] = thread(threadTest, t, size);
        thread_comm.addThreadToCommunicator(threads[t]);
    }
}
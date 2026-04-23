#include "kagen/communicators/communicator.h"
#include "kagen/communicators/communicator_interface.h"
#include "kagen/communicators/mpi_communicator.h"
#include "kagen/communicators/thread_communicator.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <typeindex>

class CommConsistencyTest : public ::testing::Test {
protected:
    std::unique_ptr<MPI_Communicator> mpi;
    std::unique_ptr<Thread_Communicator> thr;

    int rank = 0;
    int size = 1;

    void SetUp() override {
        mpi = new CommInterface(0, std::make_shared<MPI_Communicator>());

        mpi->GetWorldRank(&rank);
        mpi->GetWorldSize(&size);

        thr = std::make_unique<Thread_Communicator>(size);

        // IMPORTANT assumption
        assert(thr->pool_size() == size);
    }
};



TEST_F(CommConsistencyTest, MPI_vs_Thread_AllreduceConsistency) {
    int rank, size;
    comm->GetWorldRank(&rank);
    comm->GetWorldSize(&size);

    const int count = 8;

    std::vector<int> send(count);
    std::vector<int> mpi_recv(count, 0);
    std::vector<int> thread_recv(count, 0);

    for (int i = 0; i < count; i++) {
        send[i] = rank + 1;
    }

    MPI_Communicator mpi_comm;
    Thread_Communicator thread_comm(size); // IMPORTANT: match size

    mpi_comm.Allreduce(
        send.data(),
        mpi_recv.data(),
        count,
        typeid(int),
        CommOp::SUM
    );

    thread_comm.Allreduce(
        send.data(),
        thread_recv.data(),
        count,
        typeid(int),
        CommOp::SUM
    );

    int mismatch = 0;
    for (int i = 0; i < count; i++) {
        if (mpi_recv[i] != thread_recv[i]) {
            mismatch++;
        }
    }

    int global_mismatch = 0;
    MPI_Allreduce(&mismatch, &global_mismatch, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    EXPECT_EQ(global_mismatch, 0);
}
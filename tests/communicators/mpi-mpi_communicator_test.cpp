#include "kagen/communicators/communicator.h"
#include "kagen/communicators/communicator_interface.h"
#include "kagen/communicators/mpi_communicator.h"
#include "kagen/communicators/thread_communicator.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <typeindex>

using namespace kagen;

class MPICommTest : public ::testing::Test {
protected:
    CommInterface* comm;
    void           SetUp() override {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        comm = new CommInterface(rank, std::make_shared<MPI_Communicator>(MPI_COMM_WORLD));
        
    }

    
};
TEST_F(MPICommTest, GetWorldRankAndSize) {
    int rank = -1, size = -1;

    comm->GetRank(&rank);
    comm->GetSize(&size);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    EXPECT_EQ(rank, mpi_rank);
    EXPECT_EQ(size, mpi_size);
}

TEST_F(MPICommTest, BarrierWorks) {
    auto start = MPI_Wtime();

    comm->barrier();

    auto end = MPI_Wtime();

    EXPECT_GE(end, start);
}
TEST_F(MPICommTest, AllreduceSumInt) {
    std::vector<int> send(8);
    std::vector<int> recv(8, 0);

    int rank;
    comm->GetRank(&rank);

    for (int i = 0; i < 8; i++) {
        send[i] = rank + 1; // deterministic
    }

    comm->Allreduce(
        send.data(),
        recv.data(),
        8,
        typeid(int),
        CommOp::SUM
    );

    int expected_sum = 0;
    int size;
    comm->GetSize(&size);

    expected_sum = (size * (size + 1)) / 2; // sum of 1..size

    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(recv[i], expected_sum);
    }
}

TEST_F(MPICommTest, AllreduceInplace) {
    std::vector<int> data(10);

    int rank;
    comm->GetRank(&rank);

    for (int i = 0; i < 10; i++) {
        data[i] = rank + 1;
    }

    comm->Allreduce(
        inplace_t{},
        data.data(),
        10,
        typeid(int),
        CommOp::SUM
    );

    int size;
    comm->GetSize(&size);
    int expected = (size * (size + 1)) / 2;

    for (auto v : data) {
        EXPECT_EQ(v, expected);
    }
}

TEST_F(MPICommTest, Broadcast) {
    std::vector<int> buffer(5);

    int rank;
    comm->GetRank(&rank);

    if (rank == 0) {
        for (int i = 0; i < 5; i++) buffer[i] = i * 10;
    }

    comm->Broadcast(buffer.data(), 5, typeid(int), 0);

    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(buffer[i], i * 10);
    }
}

TEST_F(MPICommTest, Allgather) {
    int size;
    comm -> GetSize(&size);
    std::vector<int> send(3);
    std::vector<int> recv(3 * size); // will fail if more ranks → adjust in real test

    int rank;
    comm->GetRank(&rank);

    for (int i = 0; i < 3; i++) {
        send[i] = rank;
    }

    comm->Allgather(
        send.data(),
        3,
        typeid(int),
        recv.data(),
        3,
        typeid(int)
    );

    // Each rank contributes its value
    for (int r = 0; r < size; r++) {
        for (int i = 0; i < 3; i++) {
            EXPECT_EQ(recv[r * 3 + i], r);
        }
    }
}

TEST_F(MPICommTest, Alltoall) {
    int size;
    comm->GetSize(&size);

    std::vector<int> send(size);
    std::vector<int> recv(size);

    int rank;
    comm->GetRank(&rank);

    for (int i = 0; i < size; i++) {
        send[i] = rank;
    }

    comm->Alltoall(
        send.data(),
        1,
        typeid(int),
        recv.data(),
        1,
        typeid(int)
    );

    for (int i = 0; i < size; i++) {
        EXPECT_EQ(recv[i], i);
    }
}

TEST_F(MPICommTest, ExscanSum) {
    int send = 1;
    int recv = 0;

    int rank;
    comm->GetRank(&rank);

    comm->Exscan(
        &send,
        &recv,
        1,
        typeid(int),
        CommOp::SUM
    );

    if (rank == 0) {
        EXPECT_EQ(recv, 0);
    } else {
        EXPECT_EQ(recv, rank);
    }
}

TEST_F(MPICommTest, GetTimeMonotonic) {
    double t1 = comm->getTime();
    double t2 = comm->getTime();

    EXPECT_LE(t1, t2);
}

TEST_F(MPICommTest, ReduceSum) {
    int rank, size;
    comm->GetRank(&rank);
    comm->GetSize(&size);

    const int count = 8;

    std::vector<int> send(count);
    std::vector<int> recv(count, -1);

    // each rank contributes (rank + 1)
    for (int i = 0; i < count; i++) {
        send[i] = rank + 1;
    }

    int root = 0;

    comm->Reduce(
        send.data(),
        recv.data(),
        count,
        typeid(int),
        CommOp::SUM,
        root
    );

    if (rank == root) {
        int expected = 0;
        for (int r = 0; r < size; r++) {
            expected += (r + 1);
        }

        for (int i = 0; i < count; i++) {
            EXPECT_EQ(recv[i], expected);
        }
    }
}

TEST_F(MPICommTest, ReduceInplaceSum) {
    int rank, size;
    comm->GetRank(&rank);
    comm->GetSize(&size);

    const int count = 8;

    std::vector<int> data(count);

    // initialize each rank differently
    for (int i = 0; i < count; i++) {
        data[i] = 1; // simple: all ones everywhere
    }

    int root = 0;
    if(rank == root) {
        comm->Reduce(
        inplace_t{},
        data.data(),
        count,
        typeid(int),
        CommOp::SUM,
        root
    );
    } else {
        comm->Reduce(
        data.data(),
        data.data(),
        count,
        typeid(int),
        CommOp::SUM,
        root
    );
    }
    

    if (rank == root) {
        int expected = size * 1; // each rank contributed 1

        for (int i = 0; i < count; i++) {
            EXPECT_EQ(data[i], expected);
        }
    }
}
#include "kagen/communicators/communicator.h"

#include <algorithm>
#include <barrier>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

using std::thread;
using std::unordered_map;
using std::vector;

class Thread_Communicator {
private:
    static const int                       root = 0;
    vector<std::reference_wrapper<thread>> threads;
    unordered_map<std::thread::id, int>    thread_id_to_rank;

    std::vector<const void*>         shared_reduce_buffer; // Each thread writes to [rank]
    std::vector<void*>               recv_buffers;         // Each thread writes to [rank]
    std::vector<std::pair<int, int>> allgather_counts;     // For variable-length gatherings

    std::mutex              reduce_mutex;
    std::condition_variable reduce_cv;
    size_t                  threads_arrived = 0;

    template <typename T>
    std::function<void(T*, const T*, size_t)> getOp(CommOp op);
    void applyOp(CommOp op, const std::type_info& type, void* dest, const void* src, size_t count);
    void flush_buffer();
    int  getCurrentRank();

public:
    int addThreadToCommunicator(std::thread& t);

    ~Thread_Communicator();

    void GetWorldRank(int* rank) const;
    void GetWorldSize(int* size) const;

    void barrier() const;
    void abort(int code) const;

    void Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) const;

    void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) const;

    void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) const;

    void Allreduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op) const;

    void Allgather(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
        const std::type_info& recv_type) const;

    void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type) const;

    void AllgatherV(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[],
        const int displs[], const std::type_info& recv_type) const;

    void Broadcast(void* buffer, int count, const std::type_info& type, int root) const;

    void Alltoall(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
        const std::type_info& recv_type) const;

    void AlltoallV(
        const void* sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type,
        void* recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type) const;
    void   Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) const;
    void   CommitType(std::type_index type, size_t size);
    void   FreeType(std::type_index type);
    double getTime() const;
};
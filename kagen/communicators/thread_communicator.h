#include "kagen/communicators/communicator.h"
#include <thread>
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
using std::vector;
using std::unordered_map;

class Thread_Communicator : public Communicator {
    private:
        static const int                    root = 0;
        vector<std::reference_wrapper<thread>>                     threads;
        unordered_map<std::thread::id, int> thread_id_to_rank;

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
        int getCurrentRank();

    public:
    int addThreadToCommunicator(std::thread& t);

    ~Thread_Communicator() override;

    void GetWorldRank(int* rank) override;
    void GetWorldSize(int* size) override;

    void barrier() override;
    void abort(int code) override;

    void Reduce(const void* sendbuf, void* recvbuf, int count,
                const std::type_info& type, CommOp op, int root) override;

    void Reduce(inplace_t, void* recvbuf, int count,
                const std::type_info& type, CommOp op, int root) override;

    void Allreduce(const void* sendbuf, void* recvbuf, int count,
                   const std::type_info& type, CommOp op) override;

    void Allreduce(inplace_t, void* recvbuf, int count,
                   const std::type_info& type, CommOp op) override;

    void Allgather(const void* sendbuf, int sendcount, const std::type_info& send_type,
                   void* recvbuf, int recvcount, const std::type_info& recv_type) override;

    void Allgather(inplace_t, void* recvbuf, int recvcount,
                   const std::type_info& recv_type) override;

    void AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type,
                    void* recvbuf, const int recvcounts[], const int displs[],
                    const std::type_info& recv_type) override;

    void Broadcast(void* buffer, int count,
                   const std::type_info& type, int root) override;

    void Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type,
                  void* recvbuf, int recvcount, const std::type_info& recv_type) override;

    void AlltoallV(const void* sendbuf, const int sendcounts[], const int sdispls[],
                   const std::type_info& send_type,
                   void* recvbuf, const int recvcounts[], const int rdispls[],
                   const std::type_info& recv_type) override;
    void Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) override; 
    void CommitType(std::type_index type, size_t size) override;
    void FreeType(std::type_index type) override;
    double getTime() override;
};
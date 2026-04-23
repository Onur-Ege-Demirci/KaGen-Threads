#include "kagen/definitions.h"
#include "kagen/kagen.h"
#include "kagen/communicators/thread_communicator.h"

#include "communicator.h"
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

// This class is a communicator for threads within the same process, using shared memory and synchronization primitives
// to implement the communication operations. The number of threads can be increased during runtime. Decreasing isn't
// supported yet.

    
    template <typename T>
    std::function<void(T*, const T*, size_t)> Thread_Communicator::getOp(CommOp op) {
        switch (op) {
            case CommOp::SUM:
                return [](T* dest, const T* src, size_t) {
                    *dest += *src;
                };
            case CommOp::MAX:
                return [](T* dest, const T* src, size_t) {
                    *dest = std::max(*dest, *src);
                };
            case CommOp::MIN:
                return [](T* dest, const T* src, size_t) {
                    *dest = std::min(*dest, *src);
                };
            case CommOp::LOR:
                return [](T* dest, const T* src, size_t) {
                    *dest = *dest || *src;
                };
            default:
                return [](T*, const T*, size_t) {
                };
        }
    }
    unordered_map<std::type_index, size_t> type_sizes = {
        {std::type_index(typeid(int)), sizeof(int)},
        {std::type_index(typeid(double)), sizeof(double)},
        {std::type_index(typeid(unsigned int)), sizeof(unsigned int)},
        {std::type_index(typeid(long long)), sizeof(long long)},
        {std::type_index(typeid(unsigned long long)), sizeof(unsigned long long)},
        {std::type_index(typeid(long double)), sizeof(long double)}};
    // Helper to apply operation with automatic type dispatch from type_info
    static void Thread_Communicator::applyOp(CommOp op, const std::type_info& type, void* dest, const void* src, size_t count) {
        if (type == typeid(int)) {
            auto       op_func = getOp<int>(op);
            int*       d       = static_cast<int*>(dest);
            const int* s       = static_cast<const int*>(src);
            for (size_t i = 0; i < count; i++) {
                op_func(&d[i], &s[i], 1);
            }
        } else if (type == typeid(double)) {
            auto          op_func = getOp<double>(op);
            double*       d       = static_cast<double*>(dest);
            const double* s       = static_cast<const double*>(src);
            for (size_t i = 0; i < count; i++) {
                op_func(&d[i], &s[i], 1);
            }
        } else if (type == typeid(unsigned int)) {
            auto                op_func = getOp<unsigned int>(op);
            unsigned int*       d       = static_cast<unsigned int*>(dest);
            const unsigned int* s       = static_cast<const unsigned int*>(src);
            for (size_t i = 0; i < count; i++) {
                op_func(&d[i], &s[i], 1);
            }
        } else if (type == typeid(long long)) {
            auto             op_func = getOp<long long>(op);
            long long*       d       = static_cast<long long*>(dest);
            const long long* s       = static_cast<const long long*>(src);
            for (size_t i = 0; i < count; i++) {
                op_func(&d[i], &s[i], 1);
            }
        } else if (type == typeid(unsigned long long)) {
            auto                      op_func = getOp<unsigned long long>(op);
            unsigned long long*       d       = static_cast<unsigned long long*>(dest);
            const unsigned long long* s       = static_cast<const unsigned long long*>(src);
            for (size_t i = 0; i < count; i++) {
                op_func(&d[i], &s[i], 1);
            }
        } else if (type == typeid(long double)) {
            auto               op_func = getOp<long double>(op);
            long double*       d       = static_cast<long double*>(dest);
            const long double* s       = static_cast<const long double*>(src);
            for (size_t i = 0; i < count; i++) {
                op_func(&d[i], &s[i], 1);
            }
        }
    }

    void Thread_Communicator::flush_buffer() {
        for (size_t i = 0; i < shared_reduce_buffer.size(); i++) {
            shared_reduce_buffer[i] = nullptr;
        }
    }

    int Thread_Communicator::getCurrentRank() {
        auto thread_id = std::this_thread::get_id();
        return thread_id_to_rank[thread_id];
    }


    int Thread_Communicator::addThreadToCommunicator(std::thread& t) {
        int rank = threads.size();
        threads.push_back(t);
        thread_id_to_rank[t.get_id()] = rank;

        shared_reduce_buffer.emplace_back();
        recv_buffers.emplace_back();
        allgather_counts.push_back({0, 0});

        return rank;
    }

    void Thread_Communicator::GetWorldRank(int* rank) {
        auto thread_id = std::this_thread::get_id();
        *rank          = thread_id_to_rank[thread_id];
    }
    Thread_Communicator::~Thread_Communicator() {
        for (size_t i = 0; i < recv_buffers.size(); i++) {
            recv_buffers[i] = nullptr;
        }
        flush_buffer();
    }
    void Thread_Communicator::GetWorldSize(int* size) {
        *size = threads.size();
    }
    void Thread_Communicator::barrier() {
        static std::barrier b(threads.size());
        b.arrive_and_wait();
    }
    void Thread_Communicator::abort(int code) {
        std::terminate();
    }
    void
    Thread_Communicator::Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) {
        int    rank      = getCurrentRank();
        size_t elem_size = type_sizes.at(std::type_index(type));
        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            shared_reduce_buffer[rank] = sendbuf;
            threads_arrived++;

            if (rank == root) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }

                // Initialize recvbuf with root's data
                std::memcpy(recvbuf, shared_reduce_buffer[0], count * elem_size);

                // Reduce with each thread's data
                for (size_t t = 1; t < threads.size(); t++) {
                    applyOp(op, type, recvbuf, shared_reduce_buffer[t], count);
                }

                threads_arrived = 0;
                reduce_cv.notify_all();
                flush_buffer();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
            }
        }
    }

    void Thread_Communicator::Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) override {
        int    rank      = getCurrentRank();
        size_t elem_size = type_sizes.at(std::type_index(type));

        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            shared_reduce_buffer[rank] = recvbuf;
            threads_arrived++;

            if (rank == root) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }

                // Inplace reduce with each thread's data
                for (size_t t = 1; t < threads.size(); t++) {
                    applyOp(op, type, recvbuf, shared_reduce_buffer[t], count);
                }

                threads_arrived = 0;
                reduce_cv.notify_all();
                flush_buffer();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
            }
        }
    }

    void Thread_Communicator::Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) {
        int    rank      = getCurrentRank();
        size_t elem_size = type_sizes.at(std::type_index(type));

        recv_buffers[rank] = recvbuf;
        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            shared_reduce_buffer[rank] = sendbuf;
            threads_arrived++;

            if (rank == root) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }

                // Initialize recvbuf with root's data
                std::memcpy(recvbuf, shared_reduce_buffer[0], count * elem_size);

                // Reduce with each thread's data
                for (size_t t = 1; t < threads.size(); t++) {
                    applyOp(op, type, recvbuf, shared_reduce_buffer[t], count);
                }

                // Copy result to all threads' buffers
                for (size_t t = 0; t < threads.size(); t++) {
                    if (recv_buffers[t] != nullptr) {
                        std::memcpy(recv_buffers[t], recvbuf, count * elem_size);
                    }
                }
                threads_arrived = 0;
                reduce_cv.notify_all();
                flush_buffer();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                recv_buffers[rank] = nullptr;
            }
        }
    }

    // TODO_O The devil went down to Georgia....
    void Thread_Communicator::Allreduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op) {
        int    rank      = getCurrentRank();
        size_t elem_size = type_sizes.at(std::type_index(type));

        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            shared_reduce_buffer[rank] = recvbuf;
            threads_arrived++;

            if (rank == root) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }

                // In-place reduce with each thread's data
                for (size_t t = 1; t < threads.size(); t++) {
                    applyOp(op, type, recvbuf, shared_reduce_buffer[t], count);
                }

                // Copy result to all threads' buffers
                for (size_t t = 0; t < threads.size(); t++) {
                    if (recv_buffers[t] != nullptr) {
                        std::memcpy(recv_buffers[t], recvbuf, count * elem_size);
                    }
                }
                threads_arrived = 0;
                reduce_cv.notify_all();
                flush_buffer();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
            }
        }
    }
    // TODO_O The devil went down to Georgia....

    void Thread_Communicator::Allgather(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
        const std::type_info& recv_type) {
        int rank = getCurrentRank();

        size_t elem_size = type_sizes.at(std::type_index(recv_type));

        recv_buffers[rank] = recvbuf;
        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            shared_reduce_buffer[rank] = sendbuf;
            threads_arrived++;

            if (rank == root) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }

                // Gather all data into the receive buffer
                size_t offset = 0;
                for (size_t t = 0; t < threads.size(); t++) {
                    std::memcpy(
                        static_cast<uint8_t*>(recvbuf) + offset, shared_reduce_buffer[t],
                        allgather_counts[t].first * elem_size);
                    offset += allgather_counts[t].first * elem_size;
                }

                // Copy gathered result to all threads' buffers
                for (size_t t = 0; t < threads.size(); t++) {
                    if (recv_buffers[t] != nullptr) {
                        std::memcpy(recv_buffers[t], recvbuf, recvcount * elem_size);
                    }
                }
                threads_arrived = 0;
                reduce_cv.notify_all();
                flush_buffer();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                recv_buffers[rank] = nullptr;
            }
        }
    }

    void Thread_Communicator::Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type) {
        int    rank      = getCurrentRank();
        size_t elem_size = type_sizes.at(std::type_index(recv_type));

        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            shared_reduce_buffer[rank] = recvbuf;
            threads_arrived++;

            if (rank == 0) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }

                // In-place gather: each thread's data is already in recvbuf at the appropriate offset
                threads_arrived = 0;
                reduce_cv.notify_all();
                flush_buffer();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
            }
        }
    }

    void Thread_Communicator::AllgatherV(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[],
        const int displs[], const std::type_info& recv_type) {
        int    rank      = getCurrentRank();
        size_t elem_size = type_sizes.at(std::type_index(recv_type));
        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            shared_reduce_buffer[rank] = sendbuf;
            threads_arrived++;

            if (rank == root) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }

                // Gather variable-length data
                for (size_t t = 0; t < threads.size(); t++) {
                    if (recvcounts[t] > 0) {
                        std::memcpy(
                            static_cast<uint8_t*>(recvbuf) + displs[t] * elem_size, shared_reduce_buffer[t],
                            recvcounts[t] * elem_size);
                    }
                }
                threads_arrived = 0;
                reduce_cv.notify_all();
                flush_buffer();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
            }
        }
    }

    void Thread_Communicator::Broadcast(void* buffer, int count, const std::type_info& type, int root) override {
        int    rank      = getCurrentRank();
        size_t elem_size = type_sizes.at(std::type_index(type));

        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            if (rank == root) {
                shared_reduce_buffer[rank] = buffer;
            }
            threads_arrived++;

            if (rank == root) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }
                threads_arrived = 0;
                reduce_cv.notify_all();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                // Copy from root's buffer
                std::memcpy(buffer, shared_reduce_buffer[root], count * elem_size);
                threads_arrived--;
                if (threads_arrived == 0) {
                    reduce_cv.notify_all();
                }
            }
            flush_buffer();
        }
    }

    void Thread_Communicator::Alltoall(
        const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount,
        const std::type_info& recv_type) override {
        int    rank      = getCurrentRank();
        size_t elem_size = type_sizes.at(std::type_index(recv_type));
        int    msg_size  = static_cast<int>(sendcount / threads.size());

        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            shared_reduce_buffer[rank] = sendbuf;
            recv_buffers[rank]         = recvbuf;
            threads_arrived++;

            if (rank == root) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }

                // Perform all-to-all scatter
                for (size_t src = 0; src < threads.size(); src++) {
                    for (size_t dst = 0; dst < threads.size(); dst++) {
                        std::memcpy(
                            static_cast<uint8_t*>(recv_buffers[dst]) + src * msg_size * elem_size,
                            static_cast<const uint8_t*>(shared_reduce_buffer[src]) + dst * msg_size * elem_size,
                            msg_size * elem_size);
                    }
                }
                threads_arrived = 0;
                reduce_cv.notify_all();
                flush_buffer();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                recv_buffers[rank] = nullptr;
            }
        }
    }

    void Thread_Communicator::AlltoallV(
        const void* sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type,
        void* recvbuf, const int recvcounts[], const int rdispls[], const std::type_info& recv_type) override {
        int    rank      = getCurrentRank();
        size_t elem_size = type_sizes.at(std::type_index(recv_type));
        {
            std::unique_lock<std::mutex> lock(reduce_mutex);
            shared_reduce_buffer[rank] = sendbuf;
            recv_buffers[rank]         = recvbuf;
            threads_arrived++;

            if (rank == root) {
                if (threads_arrived != threads.size()) {
                    reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                }

                // Perform variable-length all-to-all scatter
                for (size_t src = 0; src < threads.size(); src++) {
                    for (size_t dst = 0; dst < threads.size(); dst++) {
                        if (sendcounts[src * threads.size() + dst] > 0) {
                            std::memcpy(
                                static_cast<uint8_t*>(recv_buffers[dst]) + rdispls[src] * elem_size,
                                static_cast<const uint8_t*>(shared_reduce_buffer[src])
                                    + sdispls[src * threads.size() + dst] * elem_size,
                                sendcounts[src * threads.size() + dst] * elem_size);
                        }
                    }
                }
                threads_arrived = 0;
                reduce_cv.notify_all();
                flush_buffer();
            } else {
                reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                recv_buffers[rank] = nullptr;
            }
        }
    }

    void Thread_Communicator::Exscan(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) override {
        //TODO_O
        return;
    } 

    void Thread_Communicator::CommitType(std::type_index type, size_t size) override {
        type_sizes[type] = size;
    }

    void Thread_Communicator::FreeType(std::type_index type) override {
        type_sizes.erase(type);
    }
    double Thread_Communicator::getTime() override {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

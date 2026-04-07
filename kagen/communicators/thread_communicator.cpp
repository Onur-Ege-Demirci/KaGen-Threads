#include "kagen.h"
#include "communicator.h"

#include <vector>
#include <thread>
#include <barrier>
#include <unordered_map>
#include <typeindex>
#include <map>
#include <mutex>
#include <condition_variable>
#include <utility>
#include <cstring>
#include <algorithm>
#include <thread>

using std::vector;
using std::thread;
static const unordered_map<std::type_index, size_t> type_sizes = {
    {typeid(int), sizeof(int)},
    {typeid(double), sizeof(double)},
    {typeid(unsigned int), sizeof(unsigned int)},
    {typeid(long long), sizeof(long long)},
    {typeid(unsigned long long), sizeof(unsigned long long)},
    {typeid(long double), sizeof(long double)} 
}; 
using std::unordered_map;

//TODO_O fix api again so it uses type_index. How much am I allowed to change it anyway? Can I get rid of the void* calls?? Does that even help?
//This class is a communicator for threads within the same process, using shared memory and synchronization primitives to implement the communication operations. The number of threads can be increased during runtime. Decreasing isn't supported (yet). 
class Thread_Communicator : Communicator {
    private:
        static const int root = 0;
        vector<thread>& threads;
        unordered_map<std::thread::id, int> thread_id_to_rank;
        
        std::vector<ConstBufferRef> shared_reduce_buffer;  // Each thread writes to [rank]
        std::vector<BufferRef> recv_buffers;               // Each thread writes to [rank]
        std::vector<std::pair<int,int>> allgather_counts;  // For variable-length gatherings

        std::mutex reduce_mutex;
        std::condition_variable reduce_cv;
        int threads_arrived = 0;
        
        template<typename T>
        static std::function<void(T*, const T*, size_t)> getOp(CommOp op) {
            switch(op){
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
                    return [](T*, const T*, size_t) {};
            }
        }
        
        // Helper to apply operation with automatic type dispatch from type_info
        static void applyOp(CommOp op, const std::type_info& type, void* dest, const void* src, size_t count) {
            if (type == typeid(int)) {
                auto op_func = getOp<int>(op);
                int* d = static_cast<int*>(dest);
                const int* s = static_cast<const int*>(src);
                for (size_t i = 0; i < count; i++) {
                    op_func(&d[i], &s[i], 1);
                }
            } else if (type == typeid(double)) {
                auto op_func = getOp<double>(op);
                double* d = static_cast<double*>(dest);
                const double* s = static_cast<const double*>(src);
                for (size_t i = 0; i < count; i++) {
                    op_func(&d[i], &s[i], 1);
                }
            } else if (type == typeid(unsigned int)) {
                auto op_func = getOp<unsigned int>(op);
                unsigned int* d = static_cast<unsigned int*>(dest);
                const unsigned int* s = static_cast<const unsigned int*>(src);
                for (size_t i = 0; i < count; i++) {
                    op_func(&d[i], &s[i], 1);
                }
            } else if (type == typeid(long long)) {
                auto op_func = getOp<long long>(op);
                long long* d = static_cast<long long*>(dest);
                const long long* s = static_cast<const long long*>(src);
                for (size_t i = 0; i < count; i++) {
                    op_func(&d[i], &s[i], 1);
                }
            } else if (type == typeid(unsigned long long)) {
                auto op_func = getOp<unsigned long long>(op);
                unsigned long long* d = static_cast<unsigned long long*>(dest);
                const unsigned long long* s = static_cast<const unsigned long long*>(src);
                for (size_t i = 0; i < count; i++) {
                    op_func(&d[i], &s[i], 1);
                }
            } else if (type == typeid(long double)) {
                auto op_func = getOp<long double>(op);
                long double* d = static_cast<long double*>(dest);
                const long double* s = static_cast<const long double*>(src);
                for (size_t i = 0; i < count; i++) {
                    op_func(&d[i], &s[i], 1);
                }
            }
        }
        
        void flush_buffer() {
            for (auto& buffer : shared_reduce_buffer) {
                buffer = ConstBufferRef();
            }
        }

        int getCurrentRank() {
            auto thread_id = std::this_thread::get_id();
            return thread_id_to_rank[thread_id];
        }
    public:

        int addThreadToCommunicator(thread& t) {
            int rank = threads.size();
            threads.push_back(t);
            thread_id_to_rank[t.get_id()] = rank;
            // Resize rank-indexed buffers
            shared_reduce_buffer.push_back(ConstBufferRef());
            recv_buffers.push_back(BufferRef());
            allgather_counts.push_back({0, 0});
            return rank;
        }

        void GetWorldRank(int *rank) override {
            auto thread_id = std::this_thread::get_id();
            *rank = thread_id_to_rank[thread_id];
        }
        ~Thread_Communicator() {
            for (auto& b : recv_buffers) { 
                b = nullptr;
            }
            //TODO_O join threads?
        }
        void GetWorldSize(int *size) override  {
            *size = threads.size();
        }
        void Barrier() override {
            static std::barrier b(threads.size());
            b.arrive_and_wait();
        }
        void Abort(int code) override {
            std::terminate();
        }
        void Reduce(ConstBufferRef sendbuf, BufferRef recvbuf, CommOp op, int root) override {
            int rank = getCurrentRank();
            size_t elem_size = type_sizes.at(std::type_index(*recvbuf.type_info));
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                shared_reduce_buffer[rank] = sendbuf;
                threads_arrived++;
                
                if (rank == root) {
                    if (threads_arrived != threads.size()) {
                        reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                    }
                    
                    // Initialize recvbuf with root's data
                    std::memcpy(recvbuf.data, shared_reduce_buffer[0].data, recvbuf.count * elem_size);
                    
                    // Reduce with each thread's data
                    for (int t = 1; t < threads.size(); t++) {
                        applyOp(op, *recvbuf.type_info, recvbuf.data, shared_reduce_buffer[t].data, recvbuf.count);
                    }
                    
                    threads_arrived = 0;
                    reduce_cv.notify_all();
                    flush_buffer();
                } else {
                    reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                }
            }
        }
        
        void Reduce(inplace_t, BufferRef recvbuf, CommOp op, int root) override {
            int rank = getCurrentRank();
            size_t elem_size = type_sizes.at(std::type_index(*recvbuf.type_info));
            
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                shared_reduce_buffer[rank] = ConstBufferRef(recvbuf.data, recvbuf.count);
                threads_arrived++;
                
                if (rank == root) {
                    if (threads_arrived != threads.size()) {
                        reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                    }
                    
                    // Inplace reduce with each thread's data
                    for (int t = 1; t < threads.size(); t++) {
                        applyOp(op, *recvbuf.type_info, recvbuf.data, shared_reduce_buffer[t].data, recvbuf.count);
                    }
                    
                    threads_arrived = 0;
                    reduce_cv.notify_all();
                    flush_buffer();
                } else {
                    reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                }
            }
        }
        void Allreduce(ConstBufferRef sendbuf, BufferRef recvbuf, CommOp op) override {
            int rank = getCurrentRank();
            size_t elem_size = type_sizes.at(std::type_index(*recvbuf.type_info));
            
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
                    std::memcpy(recvbuf.data, shared_reduce_buffer[0].data, recvbuf.count * elem_size);
                    
                    // Reduce with each thread's data
                    for (int t = 1; t < threads.size(); t++) {
                        applyOp(op, *recvbuf.type_info, recvbuf.data, shared_reduce_buffer[t].data, recvbuf.count);
                    }
                    
                    // Copy result to all threads' buffers
                    for (int t = 0; t < threads.size(); t++) {
                        if (recv_buffers[t].data != nullptr) {
                            std::memcpy(recv_buffers[t].data, recvbuf.data, recvbuf.count * elem_size);
                        }
                    }
                    threads_arrived = 0;
                    reduce_cv.notify_all();
                    flush_buffer();
                } else {
                    reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                    recv_buffers[rank] = BufferRef();
                }
            }
        }


        //TODO_O The devil went down to Georgia....
        void Allgather(ConstBufferRef sendbuf, BufferRef recvbuf, CommOp op, int root) override {
            int rank = getCurrentRank();
            size_t elem_size = type_sizes.at(std::type_index(*recvbuf.type_info));
            
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
                    for (int t = 0; t < threads.size(); t++) {
                        std::memcpy(
                            static_cast<uint8_t*>(recvbuf.data) + offset,
                            shared_reduce_buffer[t].data,
                            shared_reduce_buffer[t].count * elem_size
                        );
                        offset += shared_reduce_buffer[t].count * elem_size;
                    }
                    
                    // Copy gathered result to all threads' buffers
                    for (int t = 0; t < threads.size(); t++) {
                        if (recv_buffers[t].data != nullptr) {
                            std::memcpy(recv_buffers[t].data, recvbuf.data, recvbuf.count * elem_size);
                        }
                    }
                    threads_arrived = 0;
                    reduce_cv.notify_all();
                    flush_buffer();
                } else {
                    reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                    recv_buffers[rank] = BufferRef();
                }
            }
        }
        
        void Allgather(inplace_t, BufferRef recvbuf, CommOp op, int root) override {
            int rank = getCurrentRank();
            size_t elem_size = type_sizes.at(std::type_index(*recvbuf.type_info));
            
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                shared_reduce_buffer[rank] = ConstBufferRef(recvbuf.data, recvbuf.count);
                threads_arrived++;
                
                if (rank == root) {
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
        
        void AllgatherV(ConstBufferRef sendbuf, BufferRef recvbuf, const int recvcounts[], const int displs[]) override {
            int rank = getCurrentRank();
            size_t elem_size = type_sizes.at(std::type_index(*recvbuf.type_info));
            
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                shared_reduce_buffer[rank] = sendbuf;
                threads_arrived++;
                
                if (rank == root) {
                    if (threads_arrived != threads.size()) {
                        reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                    }
                    
                    // Gather variable-length data
                    for (int t = 0; t < threads.size(); t++) {
                        if (recvcounts[t] > 0) {
                            std::memcpy(
                                static_cast<uint8_t*>(recvbuf.data) + displs[t] * elem_size,
                                shared_reduce_buffer[t].data,
                                recvcounts[t] * elem_size
                            );
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
        
        void Broadcast(BufferRef buffer, int root) override {
            int rank = getCurrentRank();
            size_t elem_size = type_sizes.at(std::type_index(*buffer.type_info));
            
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                if (rank == root) {
                    shared_reduce_buffer[rank] = ConstBufferRef(buffer.data, buffer.count);
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
                    std::memcpy(buffer.data, shared_reduce_buffer[root].data, buffer.count * elem_size);
                    threads_arrived--;
                    if (threads_arrived == 0) {
                        reduce_cv.notify_all();
                    }
                }
                flush_buffer();
            }
        }
        
        void Alltoall(ConstBufferRef sendbuf, BufferRef recvbuf) override {
            int rank = getCurrentRank();
            size_t elem_size = type_sizes.at(std::type_index(*recvbuf.type_info));
            int msg_size = static_cast<int>(sendbuf.count / threads.size());
            
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                shared_reduce_buffer[rank] = sendbuf;
                recv_buffers[rank] = recvbuf;
                threads_arrived++;
                
                if (rank == root) {
                    if (threads_arrived != threads.size()) {
                        reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                    }
                    
                    // Perform all-to-all scatter
                    for (int src = 0; src < threads.size(); src++) {
                        for (int dst = 0; dst < threads.size(); dst++) {
                            std::memcpy(
                                static_cast<uint8_t*>(recv_buffers[dst].data) + src * msg_size * elem_size,
                                static_cast<uint8_t*>(shared_reduce_buffer[src].data) + dst * msg_size * elem_size,
                                msg_size * elem_size
                            );
                        }
                    }
                    threads_arrived = 0;
                    reduce_cv.notify_all();
                    flush_buffer();
                } else {
                    reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                    recv_buffers[rank] = BufferRef();
                }
            }
        }
        
        void AlltoallV(ConstBufferRef sendbuf, const int sendcounts[], const int sdispls[], BufferRef recvbuf, const int recvcounts[], const int rdispls[]) override {
            int rank = getCurrentRank();
            size_t elem_size = type_sizes.at(std::type_index(*recvbuf.type_info));
            
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                shared_reduce_buffer[rank] = sendbuf;
                recv_buffers[rank] = recvbuf;
                threads_arrived++;
                
                if (rank == root) {
                    if (threads_arrived != threads.size()) {
                        reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                    }
                    
                    // Perform variable-length all-to-all scatter
                    for (int src = 0; src < threads.size(); src++) {
                        for (int dst = 0; dst < threads.size(); dst++) {
                            if (sendcounts[src * threads.size() + dst] > 0) {
                                std::memcpy(
                                    static_cast<uint8_t*>(recv_buffers[dst].data) + rdispls[src] * elem_size,
                                    static_cast<uint8_t*>(shared_reduce_buffer[src].data) + sdispls[src * threads.size() + dst] * elem_size,
                                    sendcounts[src * threads.size() + dst] * elem_size
                                );
                            }
                        }
                    }
                    threads_arrived = 0;
                    reduce_cv.notify_all();
                    flush_buffer();
                } else {
                    reduce_cv.wait(lock, [&] { return threads_arrived == 0; });
                    recv_buffers[rank] = BufferRef();
                }
            }
        }

    }
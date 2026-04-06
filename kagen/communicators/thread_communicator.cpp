#include "kagen.h"
#include "communicator.h"

#include <vector>
#include <thread>
#include <barrier>
#include <unordered_map>
#include <typeindex>
#include <mutex>
#include <condition_variable>


using std::vector;
using std::thread;
using std::unordered_map;
//TODO_O flip so the threads are made by the environment
//TODO_O fix api again so it uses type_index. How much am I allowed to change it anyway? Can I get rid of the void* calls??
class Thread_Communicator : Communicator {
    private:
        static const int root = 0;
        vector<thread>& threads;
        unordered_map<std::thread::id, int> thread_id_to_rank;
        std::vector<void*> shared_reduce_buffer;  // Each thread writes to [rank]
        std::vector<void*> recv_buffers;  // Each thread writes to [rank]
        std::mutex reduce_mutex;
        std::condition_variable reduce_cv;
        static const unordered_map<std::type_index, size_t> type_sizes = {
            {typeid(int), sizeof(int)},
            {typeid(double), sizeof(double)},
            {typeid(unsigned int), sizeof(unsigned int)},
            {typeid(long long), sizeof(long long)},
            {typeid(unsigned long long), sizeof(unsigned long long)},
            {typeid(long double), sizeof(long double)}
        }; 
        int threads_arrived = 0;
        static std::function<void(const void*, const void*, size_t)> getOp(CommOp op) {   //Combines the element in the first parameter with the element in the second parameter, overwriting the first element in place. 
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
        
        void flush_buffer() {
            for (auto& buffer : shared_reduce_buffer) {
                buffer = nullptr;
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
            shared_reduce_buffer.emplace_back();
            recv_buffers.emplace_back();
            return rank;
        }

        void GetWorldRank(int *rank) override {
            auto thread_id = std::this_thread::get_id();
            *rank = thread_id_to_rank[thread_id];
        }
        ~Thread_Communicator() {
            //TODO_O join threads?
        }
        void GetWorldSize(int *size) override  {
            *size = threads.size();
        }

        //TODO_O consider what I'm going to do here
        int Execute(std::function<void(CommInterface)> func) {
            int rank;
            GetWorldRank(&rank);
            CommInterface interface(rank, this);
            return func(interface);
        }
        void Barrier() override {
            static std::barrier b(threads.size());
            b.arrive_and_wait();
        }
        void Abort(int code) override {
            std::terminate();
        }

        void Reduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) override {
            int rank = getCurrentRank();
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                shared_reduce_buffer[rank] = const_cast<void*>(sendbuf);
                threads_arrived++;
                if (rank == root) {
                    if (threads_arrived != threads.size()) {
                        reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                    }
                    
                    auto operation = getOp(op);
                    int type_size = type_sizes.at(std::type_index(type)); 
                    for (int i = 0; i < count; i++) {
                        memcpy(recvbuf + i * type_size, shared_reduce_buffer[0] + i * type_size, type_size);  // Initialize recvbuf with root's data
                        for (int t = type_size; t < threads.size(); t+= type_size) {
                            operation(recvbuf + i * type_size, shared_reduce_buffer[t] + i * type_size, type_size);  // Reduce with each thread's data
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
        void Reduce(inplace_t, void* recvbuf, int count, const std::type_info& type, CommOp op, int root) override {
            int rank = getCurrentRank();
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                shared_reduce_buffer[rank] = const_cast<void*>(recvbuf);
                threads_arrived++;
                if (rank == root) {
                    if (threads_arrived != threads.size()) {
                        reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                    }
                    
                    auto operation = getOp(op);
                    int type_size = type_sizes.at(std::type_index(type)); 
                    for (int i = 0; i < count; i++) {
                        for (int t = type_size; t < threads.size(); t+= type_size) {
                            operation(recvbuf + i * type_size, shared_reduce_buffer[t] + i * type_size, type_size);  // Reduce with each thread's data
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
        void Allreduce(const void* sendbuf, void* recvbuf, int count, const std::type_info& type, CommOp op) override {
            int rank = getCurrentRank();            
            recv_buffers[rank] = recvbuf;
            {
                std::unique_lock<std::mutex> lock(reduce_mutex);
                shared_reduce_buffer[rank] = const_cast<void*>(sendbuf);
                threads_arrived++;
                if (rank == root) {
                    if (threads_arrived != threads.size()) {
                        reduce_cv.wait(lock, [&] { return threads_arrived == threads.size(); });
                    }
                    
                    auto operation = getOp(op);
                    int type_size = type_sizes.at(std::type_index(type)); 
                    for (int i = 0; i < count; i++) {
                        memcpy(recvbuf + i * type_size, shared_reduce_buffer[0] + i * type_size, type_size);  // Initialize recvbuf with root's data
                        for (int t = type_size; t < threads.size(); t+= type_size) {
                            operation(recvbuf + i * type_size, shared_reduce_buffer[t] + i * type_size, type_size);  // Reduce with each thread's data
                        }
                    }
                    for (int t = 0; t < threads.size(); t++) {
                        if (recv_buffers[t] != nullptr) {
                            memcpy(recv_buffers[t], recvbuf, count * type_size); 
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
        void Allgather(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root) override {
            // Implement all-gather
        }
        void Allgather(inplace_t, void* recvbuf, int recvcount, const std::type_info& recv_type, CommOp op, int root) override {
            // Implement in-place all-gather
        }
        
        void AllgatherV(const void* sendbuf, int sendcount, const std::type_info& send_type, void* recvbuf, const int recvcounts[], const int displs[], const std::type_info& recv_type) override {
            // Implement all-gather-v
        }
   
        void Broadcast(void* buffer, int count, const std::type_info& type, int root) override {
            // Implement broadcast using shared memory
        }
        void Alltoall(const void* sendbuf, int sendcount, const std::type_info& send_type, void *recvbuf, int recvcount, const std::type_info& recv_type) override {
            // Implement all-to-all
        }
        void AlltoallV(const void *sendbuf, const int sendcounts[], const int sdispls[], const std::type_info& send_type, void *recvbuf, const int recvcounts[], 
            const int rdispls[], const std::type_info& recv_type) override {
            // Implement all-to-all-v
        }

    }
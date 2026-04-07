#pragma once
#include "kagen.h"
#include <functional>
#include <type_traits>

struct BufferRef {
    void* data;
    size_t count;
    const std::type_info* type_info;

    BufferRef(void* ptr = nullptr, size_t n = 0, const std::type_info* t = nullptr) 
        : data(ptr), count(n), type_info(t) {}
};

struct ConstBufferRef {
    const void* data;
    size_t count;
    const std::type_info* type_info;

    ConstBufferRef(const void* ptr = nullptr, size_t n = 0, const std::type_info* t = nullptr) 
        : data(ptr), count(n), type_info(t) {}
};

// Type-safe buffer wrapper that maintains void* compatibility
template<typename T>
struct Buffer {
    T* data;
    size_t count;
    
    Buffer(T* ptr, size_t n) : data(ptr), count(n) {}
    
    operator ConstBufferRef() const { return ConstBufferRef(data, count, &typeid(T)); }
    operator BufferRef() requires (!std::is_const_v<T>) { return BufferRef(data, count, &typeid(T)); }
    
    // Safe access methods
    T* ptr() { return data; }
    const T* ptr() const { return data; }
    size_t size() const { return count; }
};

//inplace_t is an empty class type used to indicate that reduce / allreduce should be performed inplace.
struct inplace_t {};
constexpr inplace_t inplace{}; 


enum class CommOp { 
    SUM,
    MIN,
    MAX,
    LOR
};

//TODO_O subclasses again?
class Communicator {
    public:
        virtual void GetWorldRank(int *rank) = 0;
        virtual void GetWorldSize(int *size) = 0;
        virtual void Barrier() = 0;
        virtual void Abort(int code) = 0;
        virtual void Reduce(ConstBufferRef sendbuf, BufferRef recvbuf, CommOp op, int root) = 0;
        virtual void Reduce(inplace_t, BufferRef recvbuf, CommOp op, int root) = 0;
        virtual void Allreduce(ConstBufferRef sendbuf, BufferRef recvbuf, CommOp op) = 0;
        virtual void Allgather(ConstBufferRef sendbuf, BufferRef recvbuf, CommOp op, int root) = 0;
        virtual void Allgather(inplace_t, BufferRef recvbuf, CommOp op, int root) = 0;
        virtual void AllgatherV(ConstBufferRef sendbuf, BufferRef recvbuf, const int recvcounts[], const int displs[]) = 0;
        virtual void Broadcast(BufferRef buffer, int root) = 0;
        virtual void Alltoall(ConstBufferRef sendbuf, BufferRef recvbuf) = 0;
        virtual void AlltoallV(ConstBufferRef sendbuf, const int sendcounts[], const int sdispls[], BufferRef recvbuf, const int recvcounts[], const int rdispls[]) = 0;
};


template<typename T>

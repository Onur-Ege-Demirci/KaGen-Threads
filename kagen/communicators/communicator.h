#pragma once
#include <functional>
#include <type_traits>
#include <typeindex>

//inplace_t is an empty class type used to indicate that reduce / allreduce should be performed inplace.
struct inplace_t {};
constexpr inplace_t inplace{}; 


enum class CommType {
    THREAD,
    MPI
};

enum class CommOp { 
    SUM,
    MIN,
    MAX,
    LOR
};

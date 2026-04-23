
#include "communicator_interface.h"
#include "./../kagen.h"
#include "communicator.h"
#include "mpi_communicator.h"
#include "thread_communicator.h"
#include <mpi.h>

#include <vector>
#include <thread>

using std::vector;
using std::thread;






CommInterface getMPICommInterface() {
    std::shared_ptr<Communicator> comm = std::make_shared<MPI_Communicator>();
    int rank; 
    comm -> GetWorldRank(&rank);
    CommInterface interface = CommInterface(rank, comm);
    return interface;
}

//getThreadCommunicator just constructs the communicator. The user is then responsible for creating the threads in the first place as well as lining up the relevant execution. 
//A created thread can be added to the communicator using addThreadToCommunicator, and the handle for the CommInterface received through it.
std::shared_ptr<Communicator> getThreadCommunicator() {
    auto comm = std::make_shared<Thread_Communicator>();
    return comm;
}

//Add a thread to to communicator and get the appropriate CommInterface
CommInterface addThreadToCommunicator(std::shared_ptr<Thread_Communicator> comm, thread& t) {
    int rank = comm -> addThreadToCommunicator(t);
    CommInterface interface = CommInterface(rank, comm);
    return interface;
}


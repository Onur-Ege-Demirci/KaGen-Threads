
#include "communicator_interface.h"
#include "./../kagen.h"
#include "communicator.h"


#include <vector>
#include <thread>

using std::vector;
using std::thread;

enum class CommunicatorType {
    MPI, 
    THREAD,
    HYBRID
};

//TODO_O write a header file for mpi communicator.


CommInterface getMPICommInterface() {
    auto comm = MPI_Communicator();
    int rank; 
    MPI_Comm_rank(comm, &rank);
    CommInterface interface = CommInterface(rank, comm);
    return interface;
}

//getThreadCommunicator just constructs the communicator. The user is then responsible for creating the threads in the first place as well as lining up the relevant execution. 
//A created thread can be added to the communicator using addThreadToCommunicator, and the handle for the CommInterface received through it.
Communicator getThreadCommunicator() {
    Communicator comm = Thread_Communicator();
    return comm;
}

//Add a thread to to communicator and get the appropriate CommInterface
CommInterface addThreadToCommunicator(Thread_Communicator& communicator, thread& t) {
    int rank = communicator.addThreadToCommunicator(t);
    return CommInterface(rank, communicator);
}



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
    MPI_Communicator comm = MPI_Communicator();
    int rank; 
    comm.GetWorldRank(&rank);
    CommInterface interface = CommInterface(rank, comm);
    return interface;
}

CommInterface getMPICommInterface(MPI_Comm mpi_comm) {
    MPI_Communicator comm = MPI_Communicator(mpi_comm);
    int rank; 
    comm.GetWorldRank(&rank);
    CommInterface interface = CommInterface(rank, comm);
    return interface;
}


//getThreadCommunicator just constructs the communicator. The user is then responsible for creating the threads in the first place as well as lining up the relevant execution. 
//A created thread can be added to the communicator using addThreadToCommunicator, and the handle for the CommInterface received through it.
Thread_Communicator getThreadCommunicator() {
    return Thread_Communicator();
}

//Add a thread to to communicator and get the appropriate CommInterface. 
CommInterface addThreadToCommunicator(Thread_Communicator& comm, thread& t) {
    return CommInterface(comm.addThreadToCommunicator(t), comm);
}


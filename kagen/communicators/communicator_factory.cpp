#include "communicator_interface.h"
#include "kagen.h"
#include "mpi_communicator.h"

#include <vector>
#include <thread>

using std::vector;
using std::thread;

enum class CommunicatorType {
    MPI, 
    THREAD,
    HYBRID
}


static CommInterface getCommunicator(CommunicatorType type) {

}



CommInterface getMPICommunicator() {
    Communicator comm = MPI_Communicator();
    int rank; 
    MPI_Comm_rank(&rank);
    CommInterface interface = CommInterface(rank, comm);
}


//TODO_O this needs to be called on every thread?
CommInterface getThreadCommunicator(vector<thread>& threads) {
    Communicator comm = Thread_Communicator(threads);
    
    
}
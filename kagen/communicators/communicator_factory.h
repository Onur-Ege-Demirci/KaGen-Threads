#include "kagen/communicators/communicator_interface.h"
#include "communicator.h"
#include "thread_communicator.h"
#include <thread>
CommInterface getMPICommInterface();

//getThreadCommunicator just constructs the communicator. The user is then responsible for creating the threads in the first place as well as lining up the relevant execution. 
//A created thread can be added to the communicator using addThreadToCommunicator, and the handle for the CommInterface received through it.
std::shared_ptr<Communicator> getThreadCommunicator();

//Add a thread to to communicator and get the appropriate CommInterface
CommInterface addThreadToCommunicator(std::shared_ptr<Thread_Communicator> comm, std::thread& t);


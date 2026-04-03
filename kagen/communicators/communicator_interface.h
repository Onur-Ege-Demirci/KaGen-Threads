#include "kagen.h"
#include "communicator.h"


class CommInterface {
    private:
        PEID rank;
        Communicator* comm;
    public:
       
        communicatorInterface(PEID rank, Communicator* comm) {
            this.rank = rank;
            this.comm = comm;
        }

}
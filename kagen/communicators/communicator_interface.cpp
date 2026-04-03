#include "kagen.h"
#include "communicator.h"


class CommInterface {
    private:
        PEID rank;
        Communicator comm;
    public:
        //TODO_O explicit fallthrough for every method.
        communicatorInterface(PEID rank, Communicator comm) {
            this.rank = rank;
            this.comm = comm;
        }
        int getRank() {
            return rank;
        }
        int getSize() {
            
        }


}
import sys
from LebwohlLasher_cy import main

if __name__ == '__main__':
    if int(len(sys.argv)) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        THREADS =  int(sys.argv[5])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, THREADS)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>".format(sys.argv[0]))
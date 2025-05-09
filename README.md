
## Accelerating the Lebwohl Lasher Python simulation

This project explores various methods to accelerate the Pytohn implementation of the Lebwohl
Lasher model, including NumPy vectorisation, Numba (serial and parallel), Cython (serial and parallel), and MPI. 


### How to run: 

- #### Original file

        python LebwohlLasher_og.py [ITERATIONS] [SIZE] [TEMPERATURE] [PLOTFLAG]

- #### Numpy file

        python LebwohlLasher_numpy.py [ITERATIONS] [SIZE] [TEMPERATURE] [PLOTFLAG]

- #### Numba file

        python LebwohlLasher_numba.py [ITERATIONS] [SIZE] [TEMPERATURE] [PLOTFLAG]


- #### Parallel Numba file

        python LebwohlLasher_numba_parallel.py [ITERATIONS] [SIZE] [TEMPERATURE] [PLOTFLAG] [THREADS]


- #### Serial Cython

        python setup_serial_LL.py build_ext -fi

        python run_serial_LL.py [ITERATIONS] [SIZE] [TEMPERATURE] [PLOTFLAG]


- #### Parallel Cython

        python setup_parallel_LL.py build_ext -fi

        python run_parallel_LL.py [ITERATIONS] [SIZE] [TEMPERATURE] [PLOTFLAG] [THREADS]

- #### MPI

        mpiexec -n [THREADS] python LebwohlLasher_mpi.py [ITERATIONS] [SIZE] [TEMPERATURE] [PLOTFLAG]

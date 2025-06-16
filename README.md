
## Accelerating the Lebwohl Lasher Python simulation

This project explores various methods to accelerate the Python implementation of the Lebwohl
Lasher model, including NumPy vectorisation, Numba (serial and parallel), Cython (serial and parallel), and MPI. 

The corresponding report is attached under  `Python_acceleration_report.pdf`.
Achieved Mark: 83%.

The Lebwohl-Lasher model is a Monte Carlo simulation designed to study the ordering behaviour of liquid crystals.
The Python implementation of this model used in this project simulates a two-dimensional square lattice of molecules,
which are each represented by an orientation angle. During every Monte Carlo step, each molecule experiences random
orientation changes, whereby the acceptance criteria is based on minimisatin of the energy and Boltzman probability.

However, the simulation can be very computationally expensive- especially as the problem size and iterations are
increased. This project therefore focuses on exploring different methods of accelerating the python code.


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

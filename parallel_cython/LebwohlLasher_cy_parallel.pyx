"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> (added by myself: <NREPS>)

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  (added by me: NREPS = number of times to run an experiment (the main function), in order to get a standard deviation on the runtime, for example.)
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""


import numpy as np
cimport numpy as cnp
from libc.math cimport cos, exp, M_PI
from libc.stdlib cimport rand, RAND_MAX

cimport cython
from cython.parallel cimport parallel, prange
cimport openmp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double one_energy( double[:, :] arr, int ix, int iy,int nmax) nogil:

    """
      Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      ix (int) = x lattice coordinate of cell;
      iy (int) = y lattice coordinate of cell;
        nmax (int) = side length of square lattice.
      Description:
        Function that computes the energy of a single cell of the
        lattice taking into account periodic boundaries.  Working with
        reduced energy (U/epsilon), equivalent to setting epsilon=1 in
        equation (1) in the project notes.
    Returns:
      en (float) = reduced energy of cell.
      """

    cdef: 
  
        double en = 0.0
        double ang
        int ixp = (ix+1)%nmax 
        int ixm = (ix-1)%nmax 
        int iyp = (iy+1)%nmax 
        int iym = (iy-1)%nmax 
  

    # Add together the 4 neighbour contributions
    # to the energy
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))

    return en
#=======================================================================
@cython.boundscheck(False)
def all_energy(cnp.ndarray[cnp.double_t, ndim = 2] arr_, int nmax, int threads):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """

    cdef: 
      double[:, :] arr = arr_
      double enall = 0.0
      int i, j


    for i in prange(nmax, nogil=True, num_threads = threads):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
#=======================================================================
@cython.boundscheck(False)
def get_order(cnp.ndarray[cnp.double_t, ndim = 2] arr, int nmax, int threads):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """

    cdef: 
     
      cnp.ndarray[cnp.double_t, ndim = 2] delta_ = np.eye(3,3)  

      # Generate a 3D unit vector for each cell (i,j) and put it in a (3,i,j) array. 
      cnp.ndarray[cnp.double_t, ndim = 3] lab_ = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)  
      cnp.ndarray[cnp.double_t, ndim = 2] Qab = np.zeros((3,3)) #declaring this gave a major speedup btw

      cnp.ndarray[cnp.double_t, ndim = 1] eigenvalues

      double[:, :] delta = delta_
      double[:, :, :] lab = lab_
      

      int a, b, i, j


    for a in range(3):
        for b in range(3):
            for i in prange(nmax, nogil = True, num_threads = threads):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]

    Qab = Qab/(2*nmax*nmax)
    eigenvalues = np.linalg.eigh(Qab)[0]
    return max(eigenvalues)
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def MC_step( cnp.ndarray[cnp.double_t, ndim = 2] arr_, double Ts, int nmax, int threads):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	
    Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """

    cdef: 
        double scale=0.1+Ts
        int accept = 0
    
        # Pre-compute some random numbers.  This is faster than
        # using lots of individual calls.  "scale" sets the width
        # of the distribution for the angle changes - increases
        # with temperature.

        cnp.ndarray[cnp.double_t, ndim = 2] aran_ = np.random.normal(scale=scale, size=(nmax,nmax))
        cnp.ndarray[cnp.double_t, ndim = 2] boltz_random_ = np.random.uniform(0, 1.0, size=(nmax,nmax))


        double[:, :] aran = aran_
        double[:, :] boltz_random = boltz_random_
        double[:, :] arr = arr_
  
        int i, j
        double ang, en0, en1, boltz

        double random_value



    for p in range(2):
      for i in prange(nmax, nogil=True, num_threads = threads):
          for j in range(nmax):

            if (i+j)%2 == p:

              ang = aran[i,j]

              en0 = one_energy(arr,i,j,nmax)
              arr[i, j] += ang
              en1 = one_energy(arr,i,j,nmax)

              if en1<=en0:
                  accept += 1
              else:
                  # Now apply the Monte Carlo test 
                  boltz = exp( -(en1 - en0) / Ts )

                  random_value = boltz_random[i, j]
                  if boltz >= random_value:
                      accept += 1
                  else:
                      arr[i,j] -= ang
          
              
    return accept/(nmax*nmax)

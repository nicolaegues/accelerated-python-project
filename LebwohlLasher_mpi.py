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

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI


#=======================================================================
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()  
#=======================================================================

def plotdep(energy, order, nsteps, temp): 
    
    x = np.arange(nsteps + 1)

    fig, axes = plt.subplots((2), figsize = (7, 9))
    axes[0].plot(x, energy, color = "black")
    axes[0].set_ylabel("Reduced Energy U/ε")
    axes[1].plot(x, order, color = "black")
    axes[1].set_ylabel("Order Parameter, S")

    for ax in axes: 
        ax.set_title(f"Reduced Temperature, T* = {temp}")
        ax.set_xlabel("MCS")
    
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    #plt.savefig(f"vs_MCS_plot_{current_datetime}")
    plt.show()
    
#=======================================================================

def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================

def test_equal(curr_energy): 
    og_energy = np.loadtxt("OG_output.txt", usecols=(2,))

    curr_energy = np.round(curr_energy.astype(float), 4)

    are_equal = np.array_equal(og_energy, curr_energy)

    if are_equal: 
        print("The new energy values are the same as the original energy values - all good!")
    else: 
        print("The energy values differ from the original - something went wrong. ")

#=======================================================================

def one_energy(arr,ix,iy,nmax):
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
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
#=======================================================================
def all_energy(arr,nmax, start, rows):
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
    enall = 0.0
    for i in range(start, start+rows):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
#=======================================================================

def get_order(arr,nmax):
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
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]

    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()

 
def get_order_Qab(arr,nmax, start, rows):
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
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(start, start+rows):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]

    
    return Qab
#=======================================================================
def MC_step(arr,Ts,nmax, offset, rows):
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
    scale=0.1+Ts
    accept = 0
    aran = np.random.normal(scale=scale, size=(nmax,nmax))
    for i in range(offset, offset+rows, 2):    
      for j in range(nmax):
          
          ang = aran[i,j]
          en0 = one_energy(arr,i,j,nmax)
          arr[i, j] += ang
          en1 = one_energy(arr,i,j,nmax)
          if en1<=en0:
              accept += 1
          else:
          # Now apply the Monte Carlo test - compare
          # exp( -(E_new - E_old) / T* ) >= rand(0,1)
              boltz = np.exp( -(en1 - en0) / Ts )

              if boltz >= np.random.uniform(0.0,1.0):
                  accept += 1
              else:
                  arr[i, j] -= ang

    for i in range(offset+1, offset+rows, 2):    
      for j in range(nmax):
          
          ang = aran[i,j]
          en0 = one_energy(arr,i,j,nmax)
          arr[i, j] += ang
          en1 = one_energy(arr,i,j,nmax)
          if en1<=en0:
              accept += 1
          else:
          # Now apply the Monte Carlo test - compare
          # exp( -(E_new - E_old) / T* ) >= rand(0,1)
              boltz = np.exp( -(en1 - en0) / Ts )

              if boltz >= np.random.uniform(0.0,1.0):
                  accept += 1
              else:
                  arr[i, j] -= ang

    #return accept/(nmax*nmax)
    return accept/(rows*nmax)
#=======================================================================

MAXWORKER  = 17          # maximum number of worker tasks
MINWORKER  = 1          # minimum number of worker tasks
BEGIN      = 1          # message tag
ABOVE       = 2          # message tag
BELOW       = 3          # message tag
DONE       = 4          # message tag
MASTER     = 0          # taskid of first process


def main(program, nsteps, nmax, temp, pflag, nreps):
    """
    Arguments:
    program (string) = the name of the program;
    nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
    temp (float) = reduced temperature (range 0 to 2);
    pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
  
    np.random.seed(seed=42)
    

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    numworkers = size - 1

    lattice = np.zeros((nmax, nmax))

    
    # Create array to store the runtimes
    #rep_runtimes = np.zeros(nreps)

    #for rep in range(nreps): 
          
    if rank == MASTER: 
        # Create and initialise lattice
        lattice = initdat(nmax)
        # Plot initial frame of lattice
        plotdat(lattice,pflag,nmax)

        # Create arrays to store energy, acceptance ratio and order parameter
        energy = np.zeros(nsteps+1,dtype=np.float64)
        ratio = np.zeros(nsteps+1,dtype=np.float64)
        order = np.zeros(nsteps+1,dtype=np.float64)
        # Set initial values in arrays
        energy[0] = all_energy(lattice,nmax, start=0, rows=nmax)
        ratio[0] = 0.5 # ideal value
        order[0] = get_order(lattice,nmax)

        #Distribute work to workers. Will split the lattice into row-blocks. 
        averow = nmax//numworkers
        extra = nmax%numworkers
        offset = 0

        initial = MPI.Wtime()

        #distributes the extra rows
        for i in range(1,numworkers+1):
            rows = averow
            if i <= extra:
                rows+=1

            #tell each worker who its neighbours are, since will be exchanging info. 
            #Must keep periodic boundary conditions in mind
            if i == 1:
                above = numworkers
                below = 2 if numworkers > 1 else 1
            elif i == numworkers:
                above = numworkers - 1
                below = 1
            else:
                above = i - 1
                below = i + 1

            #send startup information to each worker
            comm.send(offset, dest=i, tag=BEGIN)
            comm.send(rows, dest=i, tag=BEGIN)
            comm.send(above, dest=i, tag=BEGIN)
            comm.send(below, dest=i, tag=BEGIN)

            comm.Send(lattice[offset:offset+rows,:], dest=i, tag=BEGIN)
            offset += rows

        #wait for results from all worker tasks
        master_local_energy= np.zeros( nsteps,dtype=np.float64)
        master_local_Qab = np.zeros(( nsteps, 3, 3),dtype=np.float64)
        master_local_ratio = np.zeros( nsteps,dtype=np.float64)

        master_energy= np.zeros( nsteps,dtype=np.float64)
        master_Qab = np.zeros(( nsteps, 3, 3),dtype=np.float64)
        master_ratio = np.zeros( nsteps,dtype=np.float64)

        for i in range(1,numworkers+1):
            offset = comm.recv(source=i, tag=DONE)
            rows = comm.recv(source=i, tag=DONE)

            chunksize = nsteps

           
        comm.Reduce(master_local_energy, master_energy, op=MPI.SUM, root=MASTER)
        comm.Reduce(master_local_Qab, master_Qab, op=MPI.SUM, root=MASTER)
        comm.Reduce(master_local_ratio, master_ratio, op=MPI.SUM, root=MASTER)


        energy[1:] = master_energy
        ratio[1:] = master_ratio/numworkers #to get the mean ratios
        
        final_Qabs = master_Qab/(2*nmax*nmax)
        eigenvalues = np.linalg.eigvalsh(final_Qabs)
          
        order[1:] =  eigenvalues[:, -1]  #all the max eigenvals


        final = MPI.Wtime()
        runtime = final - initial
        #rep_runtimes[rep] = runtime

        # Final outputs
        #print("{}: Size: {:d}, Steps: {:d}, Exp. reps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Mean ratio : {:5.3f}, Time: {:8.6f} s \u00B1 {:8.6f} s".format(program, nmax,nsteps, nreps, temp,order[nsteps-1], np.mean(ratio), np.mean(rep_runtimes), np.std(rep_runtimes)))
        print("{}: Size: {:d}, Steps: {:d}, Exp. reps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Mean ratio : {:5.3f}, Time: {:8.6f} s ".format(program, nmax,nsteps, nreps, temp,order[nsteps-1], np.mean(ratio), runtime))

        # Plot final frame of lattice and generate output file
        #savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        plotdat(lattice,pflag,nmax)
        plotdep(energy, order, nsteps, temp)
        test_equal(energy)

    #************************* workers code **********************************/

    elif rank != 0: 
        
        
        offset = comm.recv(source=MASTER, tag=BEGIN)
        rows = comm.recv(source=MASTER, tag=BEGIN)
        above = comm.recv(source=MASTER, tag=BEGIN)
        below = comm.recv(source=MASTER, tag=BEGIN)

        chunksize = rows*nmax
        comm.Recv([lattice[offset, :], chunksize, MPI.DOUBLE], source=MASTER, tag=BEGIN)

        worker_energy = np.zeros(nsteps,dtype=np.float64)
        worker_ratio = np.zeros(nsteps,dtype=np.float64)
        worker_Qabs = np.zeros((nsteps, 3, 3), dtype=np.float64)

      
        for it in range(nsteps):

            worker_ratio[it] = MC_step(lattice,temp,nmax, offset, rows)
            worker_energy[it] = all_energy(lattice,nmax, offset, rows)
            worker_Qabs[it] = get_order_Qab(lattice,nmax, offset, rows)

            
            #update block with lowest row of worker above (or wrapping around to the next one)
            chunksize = nmax


         
            top_row_index    = (offset - 1) % nmax
            bottom_row_index = (offset + rows) % nmax

            # #send top row to rank 'above', receive bottom row from rank 'below'
            # comm.Sendrecv(
            #     sendbuf=lattice[offset, :], dest=above, sendtag=0,
            #     recvbuf=lattice[bottom_row_index, :], source=below, recvtag=0
            # )

            # #send bottom row to rank 'below', receive top row from rank 'above'
            # comm.Sendrecv(
            #     sendbuf=lattice[offset+rows-1, :], dest=below, sendtag=1,
            #     recvbuf=lattice[top_row_index, :], source=above, recvtag=1
            # )


            req=comm.Isend([lattice[offset+rows-1,:],chunksize,MPI.DOUBLE], dest= below, tag=ABOVE)
            req = comm.Irecv([lattice[offset-1, :], chunksize, MPI.DOUBLE], source=above, tag=ABOVE)

            req=comm.Isend([lattice[top_row_index,:],chunksize,MPI.DOUBLE], dest= above, tag=BELOW)
            req =comm.Irecv([lattice[bottom_row_index, :], chunksize, MPI.DOUBLE], source=below, tag=BELOW)



            
              
        #send arrays to master
        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(rows, dest=MASTER, tag=DONE)

        comm.Reduce(worker_energy, None, op=MPI.SUM, root=MASTER)
        comm.Reduce(worker_Qabs, None, op=MPI.SUM, root=MASTER)
        comm.Reduce(worker_ratio, None, op=MPI.SUM, root=MASTER)

        


#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
  if int(len(sys.argv)) == 6:
      PROGNAME = sys.argv[0]
      ITERATIONS = int(sys.argv[1])
      SIZE = int(sys.argv[2])
      TEMPERATURE = float(sys.argv[3])
      PLOTFLAG = int(sys.argv[4])
      NREPS =  int(sys.argv[5])
      main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, NREPS)
  else:
      print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================





import LebwohlLasher_cy_serial
import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


#threads = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


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
def plotdat(arr, pflag, nmax):
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

    u = np.cos(arr)
    v = np.sin(arr)
    cols = np.zeros((nmax,nmax))
       
    x =  np.arange(nmax, dtype = np.float64)
    y =  np.arange(nmax, dtype = np.float64)

    if pflag==0:
        return

    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = LebwohlLasher_cy_serial.one_energy(arr,i,j,nmax)
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

def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
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

def test_equal(energy): 

    og_energy = np.loadtxt("OG_output.txt", usecols=(2,))
    curr_energy = np.round(energy.astype(float), 4)

    are_equal = np.array_equal(og_energy, curr_energy)

    if are_equal: 
        print("The new energy values are the same as the original energy values - all good!")
    else: 
        print("The energy values differ from the original - something may have gone wrong. ")


#=======================================================================
def main(program, nsteps, nmax, temp, pflag):

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

    
    #cdef: 
      #double[:] rep_runtimes = np.zeros(nreps)  # Create array to store the runtimes
      #int rep
      #double[:] energy, ratio, order
      #double[:, :] lattice
      #int it



    # Create and initialise lattice
    lattice = initdat(nmax)
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1,dtype=np.float64)
    ratio = np.zeros(nsteps+1,dtype=np.float64)
    order = np.zeros(nsteps+1,dtype=np.float64)
    # Set initial values in arrays
    energy[0] = LebwohlLasher_cy_serial.all_energy(lattice,nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = LebwohlLasher_cy_serial.get_order(lattice,nmax)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = LebwohlLasher_cy_serial.MC_step(lattice,temp,nmax)
        energy[it] = LebwohlLasher_cy_serial.all_energy(lattice,nmax)
        order[it] = LebwohlLasher_cy_serial.get_order(lattice,nmax)

    final = time.time()
    runtime = final-initial

    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Mean ratio : {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps, temp,order[nsteps-1], np.mean(ratio), runtime))
    # Plot final frame of lattice and generate output file
    # savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    #plotdat(lattice,pflag,nmax)
    #plotdep(energy, order, nsteps, temp)
    #test_equal(energy)
#=======================================================================


#sizes = np.array([20, 20, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000 ])
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
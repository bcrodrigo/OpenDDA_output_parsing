import numpy as np

def orientation_average(npz_filename):
    '''
    Function that performs an average over orientations for a simulation from OpenDDA. 
    It generates a compressed .npz file with the appropriate averages.

    The .npz file will contain the following variables
        wl - wavelengths in microns (1-D array) 
        theta - Scattering Theta Angles (1-D array)
        K_oavg - Extinction Matrix averaged over orientations (Multidimensional Array)
        M_oavg - Mueller Matrix averaged over orientations (Multidimensianl Array)  
    

    Parameters
    ----------
    npz_filename : string
        Name of the npz filename (including extension) generated with read_fmatrix_rawoutput.py

    Returns
    -------
    None

    '''
    data = np.load(npz_filename)
    
    wl = data['wl']
    theta = data['theta']

    K = data['K']
    M = data['M']
    
    Nwl = len(wl)
    Ntheta = len(theta)
    
    print('\nFYI\nNwl,Nreff,Netheta,Nepsi,Nephi,Nphi,Ntheta\n')
    print(M.shape)
    
    # Perform average for MM
    #
    # Recall the MM is of shape (Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,Ntheta,4,4)
    # To perform an average over orientations, we need to:
    # 
    # 1) Select a specific wl, reff. 
    #    NOTE: for the simulations I performed, I only used 1 value of reff
    #    This reduces the dimension of the array to (Netheta,Nepsi,Nephi,Nphi,Ntheta,4,4)
    # 
    # 2) Apply the np.mean() for axis = (0,1,2,3) - this averages over all orientations as specified by 
    #    the Euler angles as well as over the Phi scattering angle. 
    #    NOTE: for the simulations I performed this angle was equivalent to the Euler Psi angle.
    #    The array is now of dimension (Ntheta,4,4)
    # 
    # 3) Save the result on M_oavg of dimension (Nwl,Ntheta,4,4)

    M_oavg = np.zeros((Nwl,Ntheta,4,4))
    
    for wl_ind in range(Nwl):

        # for every wavelength (assume a single value for reff)
        # this is always the case in the simulations I performed

        mtemp =  M[wl_ind,0,:,:,:,:,:,:,:]
        M_oavg[wl_ind,:,:,:] = mtemp.mean(axis = (0,1,2,3))
        

    # Perform Average for K matrix
    #
    # Recall that K is of shape (Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,4,4)
    # To perform an average over orientations, we need to:
    # 
    # 1) Select a specific wl, reff. 
    #    NOTE: for the simulations I performed, I only used 1 value of reff
    #    This reduces the dimension of the array to (Netheta,Nepsi,Nephi,Nphi,4,4)
    # 
    # 2) Apply the np.mean() for axis = (0,1,2,3) - this averages over all orientations as specified by 
    #    the Euler angles as well as over the Phi scattering angle. 
    #    NOTE: for the simulations I performed this angle was equivalent to the Euler Psi angle.
    #    The array is now of dimension (4,4)
    # 
    # 3) Save the result on K_oavg of dimension (Nwl,4,4)

    K_oavg = np.zeros((Nwl,4,4))

    for wl_ind in range(Nwl):

        # for every wavelength (assume a single value for reff)
        # this is always the case in the simulations I performed

        ktemp =  K[wl_ind,0,:,:,:,:,:,:]
        K_oavg[wl_ind,:,:] = ktemp.mean(axis = (0,1,2,3))


    name_no_ext = npz_filename.split('.')
    npz_filename_avg = name_no_ext[0]+'_averaged.npz'

    np.savez_compressed(npz_filename_avg, wl = wl, theta = theta,
                        M_oavg = M_oavg, K_oavg = K_oavg)
    
    print('\nCompressed File {} Saved\n'.format(npz_filename_avg))

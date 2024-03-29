import numpy as np

def calculate_pol_ellipse_params(npz_file_averaged, save_npz = False):
    '''
    Function that calculates the polarization state of scattered light, for an ensemble of 
    randomly oriented particles, for all the 'standard' incident polarization states
    as indexed below

    0 - unpolarized
    1,2- PLX, PLY (linearly polarized in x and y)
    3,4 - P45, M45 (linearly polarlized at 45 and -45 degrees)
    5,6 - RCP, LCP (right and left circularly polarized)

    Parameters
    ----------
    npz_file_averaged : String
        Name of the compressed npz file generated by `orientation_average.py`
        It's assumed to contain M_oavg (Mueller Matrix), K_oavg (Extinction Matrix),
        theta (array with scattering angle), and wl (array with wavelengths)

    save_npz : Boolean, optional
        To save a compressed npz file with all calculated parameters or not. The default is False.
        The saved parameters are also returned. 

    Returns
    -------
    all_eps : Array of size (Ninc_pol,Nwl,Ntheta)
        Ellipticity angle (epsilon) for the polarization ellipse, in degrees. 
        Note that b = a tan(epsilon)

    all_gamma : Array of size (Ninc_pol,Nwl,Ntheta)
        Tilt angle (gamma) for polarization ellipse in degrees.
        Note that gamma = 0 means the b axis of the polarization ellipse lines up with the y axis.

    all_a : Array of size (Ninc_pol,Nwl,Ntheta)
        Minor semi-axis values for polarization ellipse

    all_b : Array of size (Ninc_pol,Nwl,Ntheta)
        Major semi-axis value for polarization ellipse

    all_A : Array of size (Ninc_pol,Nwl,Ntheta)
        Amplitude of scattered light (A = sqrt(I))

    all_dop : Array of size (Ninc_pol,Nwl,Ntheta)
        Degree of polarization

    '''
    
    data = np.load(npz_file_averaged)
    theta = data['theta']
    wl = data['wl']
    M_oavg = data['M_oavg']

    # array with Stokes Vectors of 'standard' incident polarization states
    inc_pol_states = np.array([[1,0,0,0],[1,-1,0,0],[1,1,0,0],[1,0,1,0],[1,0,-1,0],[1,0,0,1],[1,0,0,-1]])
    
    Ninc_pol = 7
    Nwl = len(wl)
    Ntheta = len(theta)

    # Ellipticity Angle of polarization ellipse
    all_eps = np.zeros((Ninc_pol,Nwl,Ntheta))
    # Tilt Angle of polarization ellipse
    all_gamma = np.zeros((Ninc_pol,Nwl,Ntheta))
    
    # Minor and Major semi-axes of polarization ellipse
    all_a_values = np.zeros((Ninc_pol,Nwl,Ntheta))
    all_b_values = np.zeros((Ninc_pol,Nwl,Ntheta))

    # Amplitude of scattered light
    all_A = np.zeros((Ninc_pol,Nwl,Ntheta))

    # Degree of polarization of scattered light
    all_dop = np.zeros((Ninc_pol,Nwl,Ntheta))

    for pol_ind in range(Ninc_pol):

        # define vector with incident polarization state        
        I0 = inc_pol_states[pol_ind]
        
        for wl_ind in range(Nwl):
    
            mtemp = np.dot(M_oavg[wl_ind,:,:,:],I0)

            # Components of Stokes vector for every angle of observation
            I = mtemp[:,0]
            Q = mtemp[:,1]
            U = mtemp[:,2]
            V = mtemp[:,3]
    
            # gamma and epsilon in degrees
            temp = U/Q
            g_deg = 0.5 * np.arctan(temp) * (180.0/np.pi)
    
            # Q > 0
            ind_uq_neg = (temp < 0) & (Q > 0)
            g_deg[ind_uq_neg] = g_deg[ind_uq_neg] + 180
            # Q < 0 
            ind_q_neg = Q < 0
            g_deg[ind_q_neg] = g_deg[ind_q_neg] + 90
    

            eps = 0.5 * np.arctan(V/np.sqrt(Q**2 + U**2))
            eps_deg = eps * (180.0/np.pi)
    
            all_eps[pol_ind,wl_ind,:] = eps_deg
            all_gamma[pol_ind,wl_ind,:] = g_deg
            
            # values for polarization ellipse
            # a is in the x-direction (width for bokeh) 
            # b is in the y-direction (height for bokeh)
            tan_eps = np.tan(eps)
            # b = np.sqrt(I/(1 + (tan_eps)**2))
            # a = np.sqrt(I - b**2)
            
            # use normalized versions of a and b
            b = np.sqrt(1/(1 + (tan_eps)**2))
            a = np.sqrt(1 - b**2)

            A = np.sqrt(I)

            # Calculate degree of polarization
            dop = np.sqrt(Q**2 + U**2 + V**2)/I


            # Update multidimensional arrays 
            all_a_values[pol_ind,wl_ind,:] = a
            all_b_values[pol_ind,wl_ind,:] = b
            all_A[pol_ind,wl_ind,:] = A
            all_dop[pol_ind,wl_ind,:] = dop

    if save_npz:
        new_name = npz_file_averaged[0:-4] + '_pol_params.npz'
        np.savez_compressed(new_name, theta = theta, wl = wl,
            all_eps = all_eps, all_gamma = all_gamma, all_a = all_a_values, all_b = all_b_values,
            all_A = all_A, all_dop = all_dop)
        
        print('\nCompressed File {} Saved\n'.format(new_name))

    return all_eps,all_gamma,all_a_values,all_b_values, all_A, all_dop
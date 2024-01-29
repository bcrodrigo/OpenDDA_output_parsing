import numpy as np
import pandas as pd

def read_fmatrix_rawoutput(output_filename = ''):
    '''
    Function that reads the 'fmatrix_rawoutput.csv' output file from OpenDDA and 
    generates a compressed .npz file with multiple variables as well as the calculated
    Mueller Matrix and Extinction Matrix.
    
    The .npz file will contain the following variables
        wl - wavelengths in microns (1-D array) 
        reff - effective radii in microns (1-D array)
        etheta - Euler Theta Angles (1-D array)
        epsi - Euler Psi Angles (1-D array) 
        ephi - Euler Phi Angles (1-D array)
        phi - Scattering Phi Angles (1-D array) 
        theta - Scattering Theta Angles (1-D array)
        K - Extinction Matrix (Multidimensional Array)
        M - Mueller Matrix (Multidimensianl Array)  
    
    Parameters
    ----------
    output_filename : string, optional
        String with the name of the output .npz file, with no extension (default is empty)
        If left empty, the output npz file will be named 'fmatrix_rawoutput_parsed.npz'
    
    Returns
    -------
    None.

    '''
    # default output file name from OpenDDA
    filename = 'fmatrix_rawoutput.csv'
    
    data = pd.read_csv(filename)
    
    wl = data['wavelength'].unique()
    reff = data['effective_radius'].unique()
    euler_theta = data['euler_theta'].unique()
    euler_psi = data['euler_psi'].unique()
    euler_phi = data['euler_phi'].unique()
    sca_phi = data['sca_phi'].unique()
    sca_theta = data['sca_theta'].unique()
    
    Nwl = len(wl)
    Nreff = len(reff)
    Netheta = len(euler_theta)
    Nepsi = len(euler_psi)
    Nephi = len(euler_phi)
    Nphi = len(sca_phi)
    Ntheta = len(sca_theta)
    
    # indices for polarization states
    pol0_indices = (data['pol_state'] == 0)
    pol1_indices = (data['pol_state'] == 1)
    
    temp00 = np.array(data.loc[pol0_indices,['f_1p']].values, dtype = complex)
    temp10 = np.array(data.loc[pol0_indices,['f_2p']].values, dtype = complex)
    temp01 = np.array(data.loc[pol1_indices,['f_1p']].values, dtype = complex)
    temp11 = np.array(data.loc[pol1_indices,['f_2p']].values, dtype = complex)
    
    # Now we need to calculate the elements of the Smatrix based on fmatrix
    # 
    # $$ S_{11} = -i[f_{11} \cos\Phi+f_{12}\sin\Phi]$$
    # $$ S_{12} = i[-f_{11} \sin\Phi+f_{12}\cos\Phi]$$
    # $$ S_{21} = i[f_{21} \cos\Phi+f_{22}\sin\Phi]$$
    # $$ S_{22} = i[f_{21} \sin\Phi-f_{22}\cos\Phi]$$
    #     
    # These equations are valid because of the choice of incident polarization state
    # See my thesis [URL] Chapter XXX Section YYY

    Phivalues = data.loc[pol0_indices,['sca_phi']].values
    # convert to radians
    Phivalues = (np.pi/180.0)*Phivalues
    cos_Phi = np.cos(Phivalues)
    sin_Phi = np.sin(Phivalues)


    s00 = (-1j)*( temp00 * cos_Phi + temp01 * sin_Phi)
    s01 =  (1j)*(-temp00 * sin_Phi + temp01 * cos_Phi)
    s10 =  (1j)*( temp10 * cos_Phi + temp11 * sin_Phi)
    s11 =  (1j)*( temp10 * sin_Phi - temp11 * cos_Phi)
    
    s00 = s00.flatten()
    s01 = s01.flatten()
    s10 = s10.flatten()
    s11 = s11.flatten()
    
    Sdimension = (Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,Ntheta)
    
    # Calculate Mueller Matrix via helper function
    mm = MM_calculation(s00,s01,s10,s11,Sdimension)    
    
    
    # Take the elements of S in the forward direction
    s00_temp = s00.reshape(Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,Ntheta)
    s00_fwd = s00_temp[:,:,:,:,:,:,0]
    
    s01_temp = s01.reshape(Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,Ntheta)
    s01_fwd = s01_temp[:,:,:,:,:,:,0]
    
    s10_temp = s10.reshape(Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,Ntheta)
    s10_fwd = s10_temp[:,:,:,:,:,:,0]
    
    s11_temp = s11.reshape(Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,Ntheta)
    s11_fwd = s11_temp[:,:,:,:,:,:,0]
        
    # dimensions of S in the forward direction (scattering theta angle = 0)
    Sdim_fwd = s00_fwd.shape    
    
    K = Kmatrix_calculation(s00_fwd,s01_fwd,s10_fwd,s11_fwd,Sdim_fwd)   

    if output_filename == '':
        # remove extension
        file_noext = filename[0:-4]
        output_filename = file_noext + '_parsed.npz'

    np.savez_compressed(output_filename, wl = wl, reff = reff, 
                        etheta = euler_theta, epsi = euler_psi, ephi = euler_phi,
                        phi = sca_phi, theta = sca_theta,
                        K = K, M = mm)
    
    print('\nCompressed File {} Saved\n'.format(output_filename))


def MM_calculation(s00,s01,s10,s11,Sdimension):
    '''
    Function that calculates the Mueller Matrix based on the elements of the 
    Amplitude Scattering Matrix (S).

    Parameters
    ----------
    s00, s01, s10, s11 : 1-D complex-valued arrays
        Elements of the Amplitude Scattering Matrix
        
    Sdimension : Tuple
        Dimensions of EACH ELEMENT of the Amplitude Scattering Matrix
        namely (Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,Ntheta)

    Returns
    -------
    Mmatrix : Multi-dimensional Array
        Mueller Matrix. This will be a multidimensional array of shape 
        (Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,Ntheta,4,4)

    '''
    # Complex conjugate of Smatrix elements
    s00_conj = np.conj(s00)
    s01_conj = np.conj(s01)
    s10_conj = np.conj(s10)
    s11_conj = np.conj(s11)

    # Calculate the 16 elements of the Mueller Matrix
    # 
    # Please reference the following for further details
    # 
    #   Alexander A. Kokhanovsky. Polarization Optics of Random Media. Springer, 2003.
    #   Craig F. Bohren and Donald R. Huffman. Absorption and Scattering of Light by Small Particles. Wiley-VCH Verlag GmbH, 2007.

    m00 = 0.5*(s00 * s00_conj + s01 * s01_conj + s10 * s10_conj + s11 * s11_conj).real
    m01 = 0.5*(s00 * s00_conj - s01 * s01_conj + s10 * s10_conj - s11 * s11_conj).real 
    m02 = (s00 * s01_conj + s11 * s10_conj).real
    m03 = (s00 * s01_conj - s11 * s10_conj).imag
    
    m10 = 0.5*(s00 * s00_conj + s01 * s01_conj - s10 * s10_conj - s11 * s11_conj).real
    m11 = 0.5*(s00 * s00_conj - s01 * s01_conj - s10 * s10_conj + s11 * s11_conj).real
    m12 = (s00 * s01_conj - s11 * s10_conj).real
    m13 = (s00 * s01_conj + s11 * s10_conj).imag
    
    m20 = (s00 * s10_conj + s11 * s01_conj).real
    m21 = (s00 * s10_conj - s11 * s01_conj).real
    m22 = (s00_conj * s11 + s01 * s10_conj).real
    m23 = (s00 * s11_conj + s10 * s01_conj).imag
    
    m30 = (s00_conj * s10 + s01_conj * s11).imag
    m31 = (s00_conj * s10 - s01_conj * s11).imag
    m32 = (s11 * s00_conj - s01 * s10_conj).imag
    m33 = (s11 * s00_conj - s01 * s10_conj).real

    # reminder: tuples are immutable and can be concatenated, 
    # so the output array for this function 
    # will be Sdimension + (4,4)
    
    Mdimension = Sdimension + (4,4)
    Mmatrix = np.zeros(Mdimension)
    
    Mmatrix[:,:,:,:,:,:,:,0,0] = m00.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,0,1] = m01.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,0,2] = m02.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,0,3] = m03.reshape(Sdimension)
    
    Mmatrix[:,:,:,:,:,:,:,1,0] = m10.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,1,1] = m11.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,1,2] = m12.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,1,3] = m13.reshape(Sdimension)
    
    Mmatrix[:,:,:,:,:,:,:,2,0] = m20.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,2,1] = m21.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,2,2] = m22.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,2,3] = m23.reshape(Sdimension)
    
    Mmatrix[:,:,:,:,:,:,:,3,0] = m30.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,3,1] = m31.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,3,2] = m32.reshape(Sdimension)
    Mmatrix[:,:,:,:,:,:,:,3,3] = m33.reshape(Sdimension)
    
    return Mmatrix


def Kmatrix_calculation(s00,s01,s10,s11,Sdim_fwd):
    '''
    Function that calculates the K matrix based on the elements of the 
    Amplitude Scattering Matrix (S) in the forward direction (Sca_Theta = 0)
    

    Parameters
    ----------
    s00, s01, s10, s11 : 1-D complex-valued arrays
        Elements of the Amplitude Scattering Matrix in the forward direction (Theta = 0)

    Sdim_fwd : Tuple
        Dimensions of EACH ELEMENT of the Amplitude Scattering Matrix in the forward direction,
        namely (Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi)

    Returns
    -------
    Kmatrix : Multi-dimensional Array
        Extinction Matrix. This will be a multidimensional array of shape 
        (Nwl,Nreff,Netheta,Nepsi,Nephi,Nphi,4,4)

    '''

    # Calculate the 16 elements of the Extinction Matrix
    # 
    # Please reference the following for further details
    # 
    #   Alexander A. Kokhanovsky. Polarization Optics of Random Media. Springer, 2003.
    #   Craig F. Bohren and Donald R. Huffman. Absorption and Scattering of Light by Small Particles. Wiley-VCH Verlag GmbH, 2007.
    
    k00 = (s00 + s11).real
    k11 = k22 = k33 = k00
    
    k01 = k10 = (s00 - s11).real
    
    k02 = k20 = (s01 + s10).real
    
    k03 = k30 = ((1j)*(s01 - s10)).real
    
    k12 = (s01 - s10).real
    k21 = - k12

    k13 = ((1j)*(s01 + s10)).real
    k31 = - k13
    
    k23 = ((1j)*(s11 - s00)).real
    k32 = - k23
    
    # reminder: tuples are immutable and can be concatenated, 
    # so the output array for this function 
    # will be Sdim_fwd + (4,4)
    
    Kdimension = Sdim_fwd + (4,4)
    Kmatrix = np.zeros(Kdimension)
    
    Kmatrix[:,:,:,:,:,:,0,0] = k00.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,0,1] = k01.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,0,2] = k02.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,0,3] = k03.reshape(Sdim_fwd)

    Kmatrix[:,:,:,:,:,:,1,0] = k10.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,1,1] = k11.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,1,2] = k12.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,1,3] = k13.reshape(Sdim_fwd)
    
    Kmatrix[:,:,:,:,:,:,2,0] = k20.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,2,1] = k21.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,2,2] = k22.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,2,3] = k23.reshape(Sdim_fwd)
    
    Kmatrix[:,:,:,:,:,:,3,0] = k30.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,3,1] = k31.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,3,2] = k32.reshape(Sdim_fwd)
    Kmatrix[:,:,:,:,:,:,3,3] = k33.reshape(Sdim_fwd)
    
    return Kmatrix
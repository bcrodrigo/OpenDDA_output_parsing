# Description
Repository containing Python scripts for post-processing the simulation outputs of OpenDDA. These scripts rely on the NumPy, Pandas, and Matplotlib libraries as indicated on each script.

# Usage

The files and intended workflow are as follows:

 1. `read_fmatrix_rawoutput.py`

- Parses the simulation output *fmatrix_rawoutput.csv* file generated by OpenDDA
- Calculates the Mueller Matrix and Extinction Matrix using the `MM_calculation.py` and `Kmatrix_calculation.py` helper functions
- Saves calculated quantities as a multidimensional NumPy arrays in a compressed npz file

2. `orientation_average.py`

- Performs an average over orientations using the compressed NumPy file generated by `read_fmatrix_rawoutput.py` 
- Saves a new compressed NumPy file containing the incident wavelength, observation angle, and multidimensional NumPy arrays after averaging

3. `plot_matrix_elements.py`  
- Plots the 16 elements of the Mueller Matrix or the Extinction Matrix calculated by `orientation_average.py`
- The plots are saved as png and svg files

4. `calculate_pol_ellipse_parameters.py`

- Calculates polarization parameters using the compressed npz file generated by `orientation_average.py`
- Saves a new compressed NumPy file containing the tilt angle, ellipticity, degree of polarization, and polarization ellipse semi-axes

5. `plot_ellipse_parameters.py`

- Plots Amplitude, Degree of polarization, ellipticity and azimuth for a given incident polarization state, from the npz file generated with `calculate_pol_ellipse_parameters.py`
- The plots are saved as png and svg files

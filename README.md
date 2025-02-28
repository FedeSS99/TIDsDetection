# TIDs Detection

## Description

This project is divided in different scripts for multiple purposes with one common objetive: identify and analize TIDs (Travelling Ionospheric Disturbances) properties and related data such as local time and month ocurrence using different statistical procedures.

## Data adquisition

The data was adquired from GNSS receivers located over Mexico, which are also part of the National Seismological Service and Transboundary, Land and Atmosphere Long-Term Observational and Collaborative Network (SSN-TLALOCNet).

More specific, the numerical data used in this project consists in universal, PRN per satellite and vertical TEC (vTEC); all this read from multiple Cmn files. These said files were obtained with [_GPS-TEC analysis software version 3_](https://seemala.blogspot.com/2020/12/gps-tec-program-version-3-for-rinex-3.html).

## Software and package versions

Python was chosen for this project for the sake of "simplicity" to develop the mathematical routines using Numpy and Scipy, software provided by the Python community for scientific computing.

The Python version in which the project was made is 3.11.3.

Numpy, Scipy, Matplotlib and lmfit are the packages that were used in the elaboration of this project; to install the specific versions execute either of the following commands in a CMD/Powershell terminal in Windows or a Bash terminal in Linux:

```pip install -r requirements.txt```

In the case you use Python with Anaconda Software, use the following command:

```conda install --file requirements.txt```

## Content

The project, as already told, is divided in three different programs with different purposes:

- **Visualize_vTEC_Data.py**: This script let us choose any Cmn file, read the correspond data and visualize the original vTEC and detrended vTEC plotted against the universal time.

- **WaveletAnalaysisCMN.py**: This script was made to apply wavelet analysis to each time series of each satellite present in the input Cmn file.

  The user will be able to select a region in the power spectrum defined with 4 points by the following procedure:

  1. Using the left click with a mouse, mark a point in the spectrum. You will need 4 points to define a region of interest that could be a TID.

  2. If you want to delete the previous point, use the right click in the mouse.

  3. If there is no more regions to select, just click the mousewheel to jump to the next satellite.

  The scrip will notify you in the terminal if there is no more data to show.

- **TIDs_StatsAnalysis.py**: This script is the final step in the project since it implements different statistical procedures like boxplots, probability density functions and model fitting.

## Article
The article in which the present software was used and primarily used is show in the following link [MSTIDs article](https://www.mdpi.com/2886434).

  To do so, the script receives the paths for all "Regions" which have "Stations" folders with TIDs' output data generated with **WaveletAnalaysisCMN.py**; it will create statistical analysis for each region, station and general figures that includes all the regions.

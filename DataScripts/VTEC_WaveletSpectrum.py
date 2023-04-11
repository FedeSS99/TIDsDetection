from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, colorbar
from math import log2, pi, sqrt
import numpy as np
import scipy.signal as signal

def GetCOIcurveToPeriods(coi, dt, N):
    #Generate cone of influence given the time sampling and
    #number of data points (Time, dTEC)
    coi = coi * dt * (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    return coi

def ObtainDataFromRegion(Spectrum, Signal, TID_Region, Time, Period):
    #Obtain data from the power spectrum given the pair of
    #coordinates in TID_Region and return values like time, period
    #and power from Time, Period and Spectrum
    t0, p0 = TID_Region[0]
    t1, p1 = TID_Region[1]

    minTime, maxTime = min(t0, t1), max(t0, t1)
    minPeriod, maxPeriod = min(p0, p1), max(p0, p1)

    #Get indexes in Time and Period array given the minimum and
    #maximum respective values
    MaskTime = np.argwhere((minTime <= Time) & (Time <= maxTime))
    MaskPeriods = np.argwhere((minPeriod <= Period) & (Period <= maxPeriod))

    MaskMinTime, MaskMaxTime = MaskTime.min(), MaskTime.max()
    MaskMinPeriod, MaskMaxPeriod = MaskPeriods.min(), MaskPeriods.max()

    MaskMaxTimeIndex = MaskMaxTime + 1
    MaskMaxPeriodsIndex = MaskMaxPeriod + 1

    #Capture data from Spectrum localized inside the region of TID_Region
    RegionTID = Spectrum[MaskMinPeriod:MaskMaxPeriodsIndex, MaskMinTime:MaskMaxTimeIndex]
    maxPowerTID = RegionTID.max()
    IndexMaxPeriodTID = np.argwhere(RegionTID == maxPowerTID)
    TimeTID = Time[MaskMinTime + IndexMaxPeriodTID[0][1]]
    PeriodTID = Period[MaskMinPeriod + IndexMaxPeriodTID[0][0]]

    # Get min and max amplitude on Signal between minTime and maxTime interval
    minSig = Signal[MaskTime].min()
    maxSig = Signal[MaskTime].max()

    return TimeTID, PeriodTID, maxPowerTID, minTime, maxTime, minPeriod, maxPeriod, minSig, maxSig

def ObtainWidthAndHeight(Region):
    #Obtaining region dimensions by the following scheme:
    #           |---------------------------(R[1][0],R[1][1])
    #           |                           |
    #   (R[0][0],R[0][1])-------------------|
    Width = Region[1][0]-Region[0][0]
    Height = Region[1][1]-Region[0][1]

    return Width, Height

def CheckIntersection(Region1, Region2):
    #Check if two selected regions on the power spectrums from the
    #same satellite data intersect
    Region1Width, Region1Height = ObtainWidthAndHeight(Region1)
    Region2Width, Region2Height = ObtainWidthAndHeight(Region2)

    #Conditions to check if two rectangular regions intersect
    #in 2D space
    CondX1 = Region1[0][0] + Region1Width >= Region2[0][0]
    CondX2 = Region1[0][0] <= Region2[0][0] + Region2Width
    CondY1 = Region1[0][1] + Region1Height >= Region2[0][1]
    CondY2 = Region1[0][1] <= Region2[0][1] + Region2Height

    if CondX1 and CondX2 and CondY1 and CondY2:
        return True
    else:
        return False

def ObtainMaxMinValuesForMSTIDs(Spectrum, Periods):
    #Function to filter the total power spectrum to only watch
    #data between periods of 15 minutes and 1 hour (MSTIDs)
    MaskPeriods = np.argwhere((15.0/60.0 <= Periods) & (Periods <= 1.0))
    MaskPeriodsMin,MaskPeriodsMax = MaskPeriods.min(), MaskPeriods.max()+1

    SpectrumMSTIDs = Spectrum[MaskPeriodsMin:MaskPeriodsMax,:]
    PeriodsMSTIDs = Periods[MaskPeriodsMin:MaskPeriodsMax]
    return PeriodsMSTIDs, SpectrumMSTIDs

def CMN_Scipy_WaveletAnalysis(time_data_CMN, vtec_data_CMN, scales_j, coi_Complex, s0, fourier_factor, plot_name):
    omega0 = 6.0

    #Obtaining Wavelet coefficients by applying CWT
    Periods = fourier_factor*s0*scales_j
    waveletAmps = signal.cwt(vtec_data_CMN, signal.morlet2, scales_j, w=omega0)

    #Getting amplitude of each coefficient on waveletAmps and then
    #getting the mean of each period channel
    powerWavelet = np.abs(waveletAmps*np.conj(waveletAmps))
    Periods, powerWavelet = ObtainMaxMinValuesForMSTIDs(powerWavelet, Periods)
    MeanPowerWavelet = np.mean(powerWavelet, axis=1)

    #Create main plotting figure to use for every prn number
    MainFigure = figure(2, figsize=(10,5))
    MainFigure.subplots_adjust(hspace=0.0, wspace=0.0)
    #Configure position and width ratios of subplots
    gs = GridSpec(nrows=2, ncols=2, width_ratios=[2,1], height_ratios=[1,2])
    SubFigureSignalCMN = MainFigure.add_subplot(gs[0,0])
    SubFigureWaveletCMN = MainFigure.add_subplot(gs[1,0], sharex=SubFigureSignalCMN)
    SubFigureMeanWaveletCMN = MainFigure.add_subplot(gs[1,1])

    #Set super title of figure, text for labels and ticks parameters
    MainFigure.suptitle(f"Wavelet Analysis of .Cmn data of {plot_name}\n")
    SubFigureSignalCMN.set_ylabel("dTEC")
    SubFigureSignalCMN.tick_params(axis="x",which="both",bottom=False, top=False, labelbottom=False)
    SubFigureWaveletCMN.set_xlabel("Universal Time (U.T.)")
    SubFigureWaveletCMN.set_ylabel("Periods (Hrs)")
    SubFigureMeanWaveletCMN.set_xlabel("Mean Wavelet Power (VTEC²)")
    SubFigureMeanWaveletCMN.tick_params(axis="y",which="both",left=False, right=False, labelleft=False)

    #Extent data for plotting options
    extent = [time_data_CMN[0], time_data_CMN[-1], Periods.min(), Periods.max()]
    print(f"--Wavelet Spectrum Data of {plot_name}--")
    print(f"{s0=:f}\nInitial Time:{extent[0]:f}  Final Time:{extent[1]:f}")
    print(f"Minimum Period:{extent[2]:f}  Maximum Period:{extent[3]:f}\n")
    print(f"Time Interval of Analysis: {extent[1]-extent[0]:.3f}")

    #Plotting VTEC data
    SubFigureSignalCMN.plot(time_data_CMN, vtec_data_CMN, "k-", linewidth=1)
    SubFigureSignalCMN.set_xlim(extent[:2])

    #Plotting mesh for power values of Wavelet transform and setting options for
    #this subfigure, also plotting constant lines at 0.15Hrs and 1Hr (corresponds to
    #periodicities of MSTIDs) 
    WaveletSpectrum = SubFigureWaveletCMN.pcolormesh(time_data_CMN, Periods, powerWavelet, cmap="jet") 
    SubFigureWaveletCMN.set_ylim(*extent[2:])
    colorbar(WaveletSpectrum, ax=SubFigureMeanWaveletCMN)

    #Plotting cone of influence
    SubFigureWaveletCMN.plot(time_data_CMN, coi_Complex, "k-", linewidth=0.5)
    SubFigureWaveletCMN.fill_between(x=time_data_CMN, y1=coi_Complex, y2=extent[3], step="mid", alpha=0.25, hatch="x")
    SubFigureWaveletCMN.set_yscale("log", subs=None)
    SubFigureWaveletCMN.invert_yaxis()

    #Plotting Mean Power by each period channel
    SubFigureMeanWaveletCMN.plot(MeanPowerWavelet, Periods, "b--", linewidth=1)
    SubFigureMeanWaveletCMN.set_ylim(*extent[2:])
    SubFigureMeanWaveletCMN.set_xscale("log", subs=None)
    SubFigureMeanWaveletCMN.set_yscale("log", subs=None)
    SubFigureMeanWaveletCMN.invert_yaxis()

    MainFigure.tight_layout()

    DotsSpectrum = MainFigure.ginput(-1, timeout=0, show_clicks=True)
    TIDsRegions = []
    for i in range(0, len(DotsSpectrum)-1, 2):
        minTime, maxTime = min(DotsSpectrum[i][0], DotsSpectrum[i+1][0]), max(DotsSpectrum[i][0], DotsSpectrum[i+1][0])
        minPeriod, maxPeriod = min(DotsSpectrum[i][1], DotsSpectrum[i+1][1]), max(DotsSpectrum[i][1], DotsSpectrum[i+1][1])

        TIDsRegions.append(((minTime,minPeriod),(maxTime,maxPeriod)))

    CantRegions = len(TIDsRegions)
    DataRegions = []
    for i in range(CantRegions):
        DataRegion = ObtainDataFromRegion(powerWavelet, vtec_data_CMN, TIDsRegions[i], time_data_CMN, Periods)
        DataRegions.append(DataRegion)

    MainFigure.clear()
    return TIDsRegions, DataRegions


def CMN_WaveletAnalysis(time_data_CMN, vtec_data_CMN, dj, MainPlotName, fileOutResults):
    #Fourier Factor for Morlet
    fourier_factor_Complex = (4.0*pi)/(6 + sqrt(2 + 6**2))
    coi_Complex = fourier_factor_Complex/sqrt(2)

    #Looping through all PRN numbers to obtain their events
    #by selecting the regions of power contributions
    prnNumbers = time_data_CMN.keys()
    TIDsEvents, DataEvents = [], []
    maxValueFordTEC = 0.1
    for prn in prnNumbers:
        for interval in range(len(time_data_CMN[prn])):
            plot_name = MainPlotName + f" PRN-{prn[:-2]} Interval-{interval+1}"
            timeData, VTECdata = time_data_CMN[prn][interval], vtec_data_CMN[prn][interval]
            if VTECdata.min() <= -maxValueFordTEC or VTECdata.max() >= maxValueFordTEC:
                dt = np.diff(timeData).mean()

                Nsample = timeData.shape[0]
                s0 = 2.0*dt

                #Calculate max value for the largest scale
                J = int(log2(Nsample*dt/s0)/dj) #[Torrence & Compo, 1998] Equation 9
                #Getting scales to apply in wavelet analysis
                scales_j = 2**(np.arange(0,J+1)*dj) #[Torrence & Compo, 1998] Equation 10

                #Calculating cone of influence
                ComplexCoi_array = GetCOIcurveToPeriods(coi_Complex,dt,Nsample)

                TIDsPRN_Events, DataPRN_Events = CMN_Scipy_WaveletAnalysis(timeData, VTECdata,
                scales_j, ComplexCoi_array, s0, fourier_factor_Complex, plot_name)

                #Saving all data from regions selected in each plot
                for event, data in zip(TIDsPRN_Events, DataPRN_Events):
                    TIDsEvents.append(event)
                    DataEvents.append(data)

    #Checking if all the events are separeted TIDs, if there's an intersection
    #in the rectangular regions on the Time-Period plane then it will be saved
    #as Repeated Event
    NumEvents = len(TIDsEvents)
    RepeatedEvents = []
    for i in range(NumEvents-1):
        for j in range(i+1, NumEvents):
            Region1 = TIDsEvents[i]
            Region2 = TIDsEvents[j]
            if CheckIntersection(Region1, Region2):
                print(f"Regions {i} and {j} intersect!")
                print(f"Region{i}: ({Region1[0][0]},{Region1[1][0]}), ({Region1[0][1]},{Region1[1][1]})")
                print(f"Region{j}: ({Region2[0][0]},{Region2[1][0]}), ({Region2[0][1]},{Region2[1][1]})")

                RepeatedEvents.append(j)

    RepeatedEvents = set(RepeatedEvents)
    for i in range(NumEvents):
        if i not in RepeatedEvents:
            TimeTID, PeriodTID, maxPowerTID = DataEvents[i][:3]
            minTime, maxTime = DataEvents[i][3:5]
            minPeriod, maxPeriod = DataEvents[i][5:7]
            minSig, maxSig = DataEvents[i][7:] 
            dataString = f"{TimeTID:.6f} {PeriodTID:.6f} {maxPowerTID:.6f} {minTime:.6f} {maxTime:.6f} {minPeriod:.6f} {maxPeriod:.6f} {minSig:.6f} {maxSig:.6f}\n"
            fileOutResults.write(dataString)

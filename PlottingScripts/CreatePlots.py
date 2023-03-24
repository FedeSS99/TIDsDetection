from matplotlib import use
from matplotlib.pyplot import figure, colorbar
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

from statistics import quantiles
import numpy as np
from lmfit.models import GaussianModel

use("TkAgg")

# -------------------------------------------------------------------------------------------------------------------------------------------------
def CreateResultsFigurePower():
    #Create main figure for each analysis
    FigurePowerTime = figure(1, figsize=(8,6))
    SubPlot_PowerTime = FigurePowerTime.add_subplot(111)
    #Power-Time labels and background color
    SubPlot_PowerTime.set_xlabel("Local Time (Hours)")
    SubPlot_PowerTime.set_ylabel("TID power")

    return (FigurePowerTime, SubPlot_PowerTime)

# -------------------------------------------------------------------------------------------------------------------------------------------------
def CreateFiguresResults(Coord):
    #Create main figure for each analysis
    FigureTimeHist = figure(2, figsize=(6,6))
    FigurePeriodsHist = figure(3, figsize=(6,6))
    Figure_MonthBars = figure(4, figsize=(6,6))
    Figure_AmpsAnalysis = figure(5, figsize=(6,6))

    SubPlot_TimeHist = FigureTimeHist.add_subplot(111)
    SubPlot_PeriodHist = FigurePeriodsHist.add_subplot(111)
    SubPlot_MonthsFreq = Figure_MonthBars.add_subplot(111)
    Sub1_AmpsAnalysis = Figure_AmpsAnalysis.add_subplot(211)
    Sub2_AmpsAnalysis = Figure_AmpsAnalysis.add_subplot(212)

    #Setting titles
    Coord_String = f"\n{Coord} Region"
    SubPlot_TimeHist.set_title(Coord_String)
    SubPlot_PeriodHist.set_title(Coord_String)
    SubPlot_MonthsFreq.set_title(Coord_String)
    Figure_AmpsAnalysis.suptitle(Coord_String)

    #Time Histogram labels
    SubPlot_TimeHist.set_xlabel("Local Time (Hours)")
    SubPlot_TimeHist.set_ylabel("Month")
    #Period Histogram labels
    SubPlot_PeriodHist.set_xlabel("TID Period (Minutes)")
    SubPlot_PeriodHist.set_ylabel("Probability Density")
    #Month frequencies labels
    SubPlot_MonthsFreq.set_ylabel("Number of events")
    #Amplitudes analysis labels
    Sub1_AmpsAnalysis.set_xlabel("Local Time (Hours)")
    Sub1_AmpsAnalysis.set_ylabel("Amplitude (dTEC units)")
    Sub2_AmpsAnalysis.set_ylabel("Amplitude (dTEC units)")


    return [(FigureTimeHist, FigurePeriodsHist, Figure_MonthBars, Figure_AmpsAnalysis), 
            (SubPlot_TimeHist, SubPlot_PeriodHist, SubPlot_MonthsFreq, (Sub1_AmpsAnalysis, Sub2_AmpsAnalysis))]

# -------------------------------------------------------------------------------------------------------------------------------------------------
def SavePlot(GenName, RegName, PlotFigure):
    for format in ["png", "pdf"]:
        PlotFigure.savefig(f"./../Results/{RegName}/{GenName}{RegName}.{format}")

# -------------------------------------------------------------------------------------------------------------------------------------------------
def Add_TimePowerDataResultsToPlot(Time, Power, Plots, Color, Start):
    #Plotting boxplots for each hour interval given the station's index
    Indexes = list(range(Start, 24, 3))
    for Index in Indexes:
        
        # Creating mask of Time array given a one hour interval
        MaskTime = np.where((Index <= Time) & (Time <= Index+1), True, False)
        PowerMask = Power[MaskTime]
        PowerMask = PowerMask.reshape(PowerMask.size, 1)
        
        # Create a boxplot only if the size of the Power mask array has elements
        if PowerMask.size  > 0:

            BoxPlot = Plots[1].boxplot(PowerMask, sym="x", positions=[Index + 0.5], patch_artist=True,
                                        widths=0.25)
            
            # Change colors of the boxplot given Color input
            for ComponentBoxPlot in [BoxPlot["whiskers"], BoxPlot["caps"], BoxPlot["fliers"], BoxPlot["medians"]]:
                for patch in ComponentBoxPlot:
                    patch.set_color(Color)
                    patch.set_linewidth(2)

            for BoxComponent in BoxPlot["boxes"]:
                BoxComponent.set_facecolor("None")
                BoxComponent.set_edgecolor(Color)

    Plots[1].set_xticks([])
    XTICKS = [i for i in range(0, 25, 4)]
    Plots[1].set_xticks(ticks=XTICKS, labels=XTICKS)

    return BoxPlot["boxes"][0]

# -------------------------------------------------------------------------------------------------------------------------------------------------
def Add_TimeMonthsHistogramToPlot(HistogramMonths, absMin, absMax, Plots, Name):
    #Setting number of bins and time range for histogram
    TimeRange = (0.0, 24.0)
    MonthRange = (0, 12)
    #Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = np.linspace(0.5,11.5,12,endpoint=True)
    Plots[1][0].set_yticks(MonthAxisData, MonthTicks)

    # Set the limits for Local Time and indexes for each Month
    extent = (*TimeRange,*MonthRange)
    timeTicks = np.arange(0, 25, 6)
    Plots[1][0].set_xlim(*TimeRange)
    Plots[1][0].set_ylim(*MonthRange)
    Plots[1][0].set_xticks(timeTicks)

    # Define the colormap
    Cmap = get_cmap("viridis")
    # Extract all colors from the jet map
    Cmaplist = [Cmap(i) for i in range(Cmap.N)]
    # Force the first color entry to be transparent
    Cmaplist[0] = (0, 0, 0, 0.0)

    # Create the new map
    Cmap = LinearSegmentedColormap.from_list('Ocurrence Map', Cmaplist, Cmap.N)

    # define the bins and normalize
    bounds = np.linspace(absMin, absMax, 7)
    Norm = BoundaryNorm(bounds, Cmap.N)

    HistogramaImagen = Plots[1][0].imshow(HistogramMonths, cmap=Cmap, norm=Norm,
    interpolation="spline16", aspect="auto", origin="lower", extent=extent)
    colorbar(HistogramaImagen, ax=Plots[1][0], label="% Ocurrence")

    #Extract terminator hours
    if "Center" in Name:
        TerminatorsFile = "TerminatorHours_Center1.dat"
    elif "North" in Name:
        TerminatorsFile = "TerminatorHours_North.dat"
    elif "South" in Name:
        TerminatorsFile = "TerminatorHours_South.dat"

    RiseHours, SetHours = np.loadtxt(TerminatorsFile, dtype=np.float64,
    usecols=(1, 2), unpack=True, skiprows=1)
    NumMonthTerminator = np.linspace(0.0, 12.0, RiseHours.size)
    Plots[1][0].plot(RiseHours, NumMonthTerminator, "--k", linewidth=1.0)
    Plots[1][0].plot(SetHours, NumMonthTerminator, "--k", linewidth=1.0)

    Plots[0][0].tight_layout()
    SavePlot("OcurrenceTIDs_", Name, Plots[0][0])

# -------------------------------------------------------------------------------------------------------------------------------------------------
Gaussian_Dist = lambda x, A, sig, mu: (A/(sig*(2.0*np.pi)**0.5))*np.exp(-0.5*((x-mu)/sig)**2.0)
def Add_PeriodHistogramToPlot(Period, Time_TIDs, Months_TIDs, Plots, Name):
    
    Period = 60.0*Period
    #Extract terminator hours
    if "Center" in Name:
        TerminatorsFile = "TerminatorHours_Center1.dat"
    elif "North" in Name:
        TerminatorsFile = "TerminatorHours_North.dat"
    elif "South" in Name:
        TerminatorsFile = "TerminatorHours_South.dat"

    RiseHours, SetHours = np.loadtxt(TerminatorsFile, dtype=np.float64,
    usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:DivH_12], SetHours[0:SizeData:DivH_12]

    DayTIDsPeriods = []
    NightTIDsPeriods = []
    for month in range(1,13):
        Conds_month = Months_TIDs==month
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            Period_Conds_month = Period[Conds_month]
            MaskDay = (RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1])
            MaskNight = ~MaskDay

            DayTIDsPeriods.append(Period_Conds_month[MaskDay])
            NightTIDsPeriods.append(Period_Conds_month[MaskNight])

    DayTIDsPeriods = np.concatenate(tuple(DayTIDsPeriods))
    NightTIDsPeriods = np.concatenate(tuple(NightTIDsPeriods))
    for PeriodData, Color, NamePlot in zip([DayTIDsPeriods, NightTIDsPeriods], ["tab:orange", "tab:blue"], ["Day", "Night"]):
        #Setting number of bins, period range and also the width of the bars
        Quantiles = quantiles(PeriodData, n=4)
        h = 2.0*(Quantiles[2]-Quantiles[0])*(PeriodData.size**(-1/3))
        PeriodRange = (PeriodData.min(), PeriodData.max())
        PeriodBins = int((PeriodRange[1]-PeriodRange[0])/h)
        BarsPeriod = np.linspace(PeriodRange[0], PeriodRange[1], PeriodBins)
        Width = np.diff(BarsPeriod).mean()

        #Obtaining histogram from period data with given bins and range
        PeriodHistogram, Edges = np.histogram(PeriodData, bins=PeriodBins, range=PeriodRange,
                                              density=True)

        #Getting mean, deviation of period data and max value of Ocurrence
        Mean, Deviation = PeriodData.mean(), PeriodData.std()
        MaxValue = PeriodHistogram.max()

        #Declaring an Exponential Gaussian Model as the proposed theorical distribution
        GaussianToFit = GaussianModel()
        #Setting parameters
        ParametersExpGaussian = GaussianToFit.make_params(amplitude=MaxValue,
        center=Mean, sigma=Deviation)
        #Calculate best fit
        ExpGaussianFitResult = GaussianToFit.fit(PeriodHistogram, ParametersExpGaussian, x=Edges[1:])

        ParamsResults = ExpGaussianFitResult.params
        AmpFit = ParamsResults["amplitude"].value
        MeanFit, MeanError = ParamsResults["center"].value, ParamsResults["center"].stderr
        SigmaFit, SigmaError = ParamsResults["sigma"].value, ParamsResults["sigma"].stderr

        PeriodRange = np.linspace(0.0, max(Edges), 100)
        labelFit = NamePlot+"\n"+r"$\mu$={0:.3f}$\pm${1:.3f}".format(MeanFit,MeanError)+"\n"+r"$\sigma$={0:.3f}$\pm${1:.3f}".format(SigmaFit,SigmaError)

        Plots[1][1].bar(BarsPeriod, height=PeriodHistogram, width=Width, align="edge",
        facecolor=Color, edgecolor=Color, alpha=0.25)
        Plots[1][1].plot(PeriodRange, Gaussian_Dist(PeriodRange,AmpFit,SigmaFit,MeanFit), linestyle="--", color=Color, linewidth=1.5,
        label=labelFit)

    Plots[1][1].legend()
    Plots[0][1].tight_layout()
    SavePlot("PeriodDistribution_", Name, Plots[0][1])

# -------------------------------------------------------------------------------------------------------------------------------------------------
def Add_BarsFreq_Month(Time_TIDs, Months_TIDs, Plots, Name):
    #Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = np.linspace(0.5,11.5,12,endpoint=True)
    Plots[1][2].set_xticks(MonthAxisData, MonthTicks)

    #Extract terminator hours
    if "Center" in Name:
        TerminatorsFile = "TerminatorHours_Center1.dat"
    elif "North" in Name:
        TerminatorsFile = "TerminatorHours_North.dat"
    elif "South" in Name:
        TerminatorsFile = "TerminatorHours_South.dat"

    RiseHours, SetHours = np.loadtxt(TerminatorsFile, dtype=np.float64,
    usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:DivH_12], SetHours[0:SizeData:DivH_12]

    NumEventerPerMonth = []
    for month in range(1,13):
        Conds_month = Months_TIDs==month
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            NumDayNight_month = np.where((RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1]), 1, 0)
            NumDay = (NumDayNight_month==1).sum()
            NumNight = (NumDayNight_month==0).sum()

            NumEventerPerMonth.append((month, NumDay, NumNight))
        else:
            NumEventerPerMonth.append((month, 0, 0))

    NumEventerPerMonth.sort(key = lambda e: e[0])
    _, NumEvents_Day, NumEvents_Night = zip(*NumEventerPerMonth)

    Plots[1][2].bar(x=MonthAxisData - 0.25, height=NumEvents_Day, width=0.25,
    align="edge", edgecolor="k", facecolor="r", label="Day")
    Plots[1][2].bar(x=MonthAxisData, height=NumEvents_Night, width=0.25,
    align="edge", edgecolor="k", facecolor="b", label="Night")

    Plots[1][2].legend()
    Plots[0][2].tight_layout()
    SavePlot("DayNightTIDs_", Name, Plots[0][2])

# -------------------------------------------------------------------------------------------------------------------------------------------------
def Add_AmplitudesAnalysis(MinA, MaxA, Time_TIDs, Months_TIDs, Plots, Name):
    #Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    #MonthTicks = ["Jan", "Mar", "May", "Jul", "Sep", "Nov"]
    MonthAxisData = np.linspace(0.5,11.5,12,endpoint=True)
    Plots[1][3][1].set_xticks(MonthAxisData, MonthTicks)

    #Extract terminator hours
    if "Center" in Name:
        TerminatorsFile = "TerminatorHours_Center1.dat"
    elif "North" in Name:
        TerminatorsFile = "TerminatorHours_North.dat"
    elif "South" in Name:
        TerminatorsFile = "TerminatorHours_South.dat"

    RiseHours, SetHours = np.loadtxt(TerminatorsFile, dtype=np.float64,
    usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:DivH_12], SetHours[0:SizeData:DivH_12]

    # START ANALYSIS BY MONTHS DIVIDED IN DAY AND NIGHT ACTIVITY
    MeanAmps_Dict = dict(Day=dict(Means=[], STD=[]), Night=dict(Means=[], STD=[]))
    for month in range(1,13,1):
        Conds_month = Months_TIDs==month
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            MinA_Conds_month = MinA[Conds_month]
            MaxA_Conds_month = MaxA[Conds_month]

            MaskDay = (RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1])
            MaskNight = ~MaskDay

            # Mean and Std of average amplitud from absolut min and max amplitude for...

            # day and ...
            AverageMinMax_DayAmps = (np.abs(MinA_Conds_month[MaskDay]) + MaxA_Conds_month[MaskDay])/2.0
            Mean_DayAverageAmps = AverageMinMax_DayAmps.mean()
            STD_DayAverageAmps = AverageMinMax_DayAmps.std()

            # night
            AverageMinMax_NightAmps = (np.abs(MinA_Conds_month[MaskNight]) + MaxA_Conds_month[MaskNight])/2.0
            Mean_NightAverageAmps = AverageMinMax_NightAmps.mean()
            STD_NightAverageAmps = AverageMinMax_NightAmps.std()

            MeanAmps_Dict["Day"]["Means"].append( Mean_DayAverageAmps )
            MeanAmps_Dict["Day"]["STD"].append( STD_DayAverageAmps )
            MeanAmps_Dict["Night"]["Means"].append( Mean_NightAverageAmps )
            MeanAmps_Dict["Night"]["STD"].append( STD_NightAverageAmps )

        else:
            MeanAmps_Dict["Day"]["Means"].append( 0 )
            MeanAmps_Dict["Day"]["STD"].append( 0 )
            MeanAmps_Dict["Night"]["Means"].append( 0 )
            MeanAmps_Dict["Night"]["STD"].append( 0 )

    # START ANALYSIS GIVEN THE ACTIVITY IN LOCAL TIME
    Indexes = list(range(0,24,2))
    for Index in Indexes:
        MaskTime = np.where((Index <= Time_TIDs) & (Time_TIDs <= Index+2), True, False)
        AverageMinMax_Amps = (np.abs(MinA[MaskTime]) + MaxA[MaskTime])/2.0
        AverageMinMax_Amps = AverageMinMax_Amps.reshape(AverageMinMax_Amps.size, 1)
        if AverageMinMax_Amps.size  > 0:

            BoxPlot = Plots[1][3][0].boxplot(AverageMinMax_Amps, sym="x", positions=[Index + 0.5], patch_artist=True,
                                        widths=0.5)
            
            for ComponentBoxPlot in [BoxPlot["whiskers"], BoxPlot["caps"], BoxPlot["fliers"], BoxPlot["medians"]]:
                for patch in ComponentBoxPlot:
                    patch.set_color("k")
                    patch.set_linewidth(2)

            for BoxComponent in BoxPlot["boxes"]:
                BoxComponent.set_facecolor("None")
                BoxComponent.set_edgecolor("k")

    Plots[1][3][0].set_xticks([])
    XTICKS = [i for i in range(0, 25, 4)]
    Plots[1][3][0].set_xticks(ticks=XTICKS, labels=XTICKS)


    Colors_Errobars = ("lime", "purple")
    N_plot = 0
    for KeyDict, phaseErrorPlot, Color in zip(MeanAmps_Dict.keys(), (-.2, .2), Colors_Errobars):
        Dict_By_DayNight = MeanAmps_Dict[KeyDict]

        Dict_By_DayNight["Means"] = np.array(Dict_By_DayNight["Means"])
        Dict_By_DayNight["STD"] = np.array(Dict_By_DayNight["STD"])

        Y = Dict_By_DayNight["Means"]
        Yerr = Dict_By_DayNight["STD"]
        MaskNonZeros = Y != 0
            
        Plots[1][3][1].errorbar(x=MonthAxisData[MaskNonZeros] + phaseErrorPlot, y=Y[MaskNonZeros], yerr=Yerr[MaskNonZeros],
                                capsize=5, capthick=1, elinewidth=1, ecolor=Color, color=Color, 
                                label=KeyDict, fmt="o")
                                

    Plots[1][3][1].legend()
    Plots[0][3].tight_layout()
    SavePlot("AmpsAnalysis_", Name, Plots[0][3])
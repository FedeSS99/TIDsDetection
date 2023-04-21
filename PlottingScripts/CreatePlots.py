from matplotlib import use
from matplotlib.pyplot import figure, subplots, colorbar
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

from scipy.stats.mstats import mquantiles
import numpy as np
from lmfit.models import GaussianModel, PowerLawModel

use("TkAgg")

# Dictionary to extract filename of Terminator data for each region
TerminatorsDict = dict(
    North="./TerminatorData/TerminatorHours_North.dat",
    Center="./TerminatorData/TerminatorHours_Center.dat",
    South="./TerminatorData/TerminatorHours_South.dat"
    )

# Dictionary to extract colors to use as day-night filter for amplitude-power data
AmpPower_COLORS = dict(
    North = ("#1100FF", "#EEFF00"),
    Center = ("#00FF07", "#FF00F8"),
    South = ("#FF0001", "#00FFFE")
)

# -------------------------------------------------------------------------------------------------------------------------------------------------
def CreateFigureTimePower():
    #Create main figure
    FigurePowerTime = figure(1, figsize=(8,6))
    SubPlot_PowerTime = FigurePowerTime.add_subplot(111)
    #Power-Time labels
    SubPlot_PowerTime.set_xlabel("Local Time (Hours)")
    SubPlot_PowerTime.set_ylabel("TID power")

    return (FigurePowerTime, SubPlot_PowerTime)

def CreateFigureAmplitudePower(Nplots):
    #Create main figure
    FigureAmplitudePower, Subplots = subplots(num=2, nrows=Nplots, ncols=1, sharex=True, figsize=(6,6))
    for i in range(Nplots):
        # Amplitude-Power labels
        Subplots[i].set_ylabel("TID power")

    Subplots[Nplots-1].set_xlabel("Average absolute amplitude (dTEC)")
    FigureAmplitudePower.subplots_adjust(hspace=0.0)

    return (FigureAmplitudePower, Subplots)

# -------------------------------------------------------------------------------------------------------------------------------------------------
def CreateFiguresResults(Name, Stat_or_Reg):
    #Create main figure for each analysis
    FigureTimeHist = figure(3, figsize=(6,6))
    FigurePeriodsHist = figure(4, figsize=(6,6))
    Figure_MonthBars = figure(5, figsize=(6,6))
    Figure_AmpsAnalysis = figure(6, figsize=(6,6))

    SubPlot_TimeHist = FigureTimeHist.add_subplot(111)
    SubPlot_PeriodHist = FigurePeriodsHist.add_subplot(111)
    SubPlot_MonthsFreq = Figure_MonthBars.add_subplot(111)
    Sub1_AmpsAnalysis = Figure_AmpsAnalysis.add_subplot(211)
    Sub2_AmpsAnalysis = Figure_AmpsAnalysis.add_subplot(212)

    #Setting titles
    if Stat_or_Reg == "Stat":
        Coord_String = f"\n{Name} Station"
    elif Stat_or_Reg == "Reg":
        Coord_String = f"\n{Name} Region"

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
def SaveRegionPlot(GenName, RegName, PlotFigure):
    for format in ["png", "pdf"]:
        PlotFigure.savefig(f"./../Results/{RegName}/{GenName}{RegName}.{format}")

def SaveStationPlot(GenName, RegName, StationName, PlotFigure):
    for format in ["png", "pdf"]:
        PlotFigure.savefig(f"./../Results/{RegName}/{StationName}/{GenName}{StationName}.{format}")

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
# Same power law function as used in lmfit module
# check PowerLawModel in https://lmfit.github.io/lmfit-py/builtin_models.html
PowerFunction = lambda x, A, k: A*(x**k)

def Add_AmplitudePowerScatterPlot(MinA, MaxA, Power, Time, Months, Plots, Marker, Index, RegionName):
    # Obtain average absolute amplitude
    AverageAmplitude = 0.5*(np.abs(MinA) + MaxA)
    Arg_AveAmpSort = np.argsort(AverageAmplitude)
    AverageAmplitude = AverageAmplitude[Arg_AveAmpSort]
    Power = Power[Arg_AveAmpSort]
    Time = Time[Arg_AveAmpSort]
    Months = Months[Arg_AveAmpSort]

    # Start a PowerLaw object
    PowerModel = PowerLawModel()
    # Define initial guessing values
    Params = PowerModel.guess(Power, x=AverageAmplitude)
    # Input average absoluted amplitude and TIDs'power as independent and dependent variables
    PowerModelFit = PowerModel.fit(Power, Params, x=AverageAmplitude)

    # Extract amplitude and power of best fit
    BestPowerFitParams = PowerModelFit.params
    Best_A = BestPowerFitParams["amplitude"].value
    Best_k = BestPowerFitParams["exponent"].value
    
    # Get R2 score of the best fit
    R2_Score = PowerModelFit.rsquared

    # Get best fit of the amplitude-power model
    Best_AmpPowerFit = PowerModelFit.best_fit

    print(f"--Power Law Model for Amplitude-Power plot--\nAmplitude = {Best_A:.3f}\nExponent = {Best_k:.3f}\nR2-Score = {R2_Score:.3f}\n")

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
    usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:DivH_12], SetHours[0:SizeData:DivH_12]

    # Apply day-night filter for amplitude-power data
    DayNightAmplitude = dict(DAY=[], NIGHT=[])
    DayNightPower = dict(DAY=[], NIGHT=[])
    NumDay, NumNight = 0, 0
    for month in range(1,13):
        # Separate data by given month
        Conds_month = Months==month

        # Check if there is any data point within this month
        if Conds_month.any():

            # Create arrays given the month and the time to separate in day and night
            Time_Conds_month = Time[Conds_month]
            AveAmplitude_Conds_month = AverageAmplitude[Conds_month]
            Power_Conds_month = Power[Conds_month]


            # Day and night masks and count the total of
            MaskDay = (RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1])
            MaskNight = ~MaskDay

            NumDay += MaskDay.sum()
            NumNight += MaskNight.sum()

            DayNightAmplitude["DAY"].append(AveAmplitude_Conds_month[MaskDay])
            DayNightAmplitude["NIGHT"].append(AveAmplitude_Conds_month[MaskNight])
            DayNightPower["DAY"].append(Power_Conds_month[MaskDay])
            DayNightPower["NIGHT"].append(Power_Conds_month[MaskNight])

    DayNightAmplitude["DAY"] = np.concatenate(tuple(DayNightAmplitude["DAY"]))
    DayNightAmplitude["NIGHT"] = np.concatenate(tuple(DayNightAmplitude["NIGHT"]))
    DayNightPower["DAY"] = np.concatenate(tuple(DayNightPower["DAY"]))
    DayNightPower["NIGHT"] = np.concatenate(tuple(DayNightPower["NIGHT"]))

    # Add scatter plot of average amplitudes and power by day-night filter
    for moment, color in zip(["DAY", "NIGHT"], AmpPower_COLORS[RegionName]):

        Plots[1][Index].scatter(DayNightAmplitude[moment], DayNightPower[moment], alpha=0.25,
                         c=color, marker=Marker, label= RegionName + "-" + moment)
        Plots[1][Index].set_yscale("log", subs=None)
    
    # Add best fit of power law model for average amplitudes and power data
    Plots[1][Index].plot(AverageAmplitude, Best_AmpPowerFit, "--k")
    Plots[1][Index].text(0.05, 0.95, f"Amplitude = {Best_A:.3f}\nExponent = {Best_k:.3f}\n"+r"$R^{{2}}$ = {0:.3f}".format(R2_Score),
                         horizontalalignment = "left", verticalalignment = "top", fontsize = 10, 
                         transform = Plots[1][Index].transAxes)

    # And finally, add number of day and night events
    Plots[1][Index].text(0.95, 0.05, f"Day = {NumDay} Night = {NumNight}",
                         horizontalalignment = "right", verticalalignment = "bottom", fontsize = 8, 
                         transform = Plots[1][Index].transAxes)

# -------------------------------------------------------------------------------------------------------------------------------------------------
def ObtainCMAPandNORM(OcurrenceArray):
    # Define the colormap
    CMAP = get_cmap("jet")
    # Extract all colors from the jet map
    CMAPlist = [CMAP(i) for i in range(CMAP.N)]
    # Force the first color entry to be transparent
    CMAPlist[0] = (0.0, 0.0, 0.0, 0.0)

    # Create the new CMAP
    CMAP = LinearSegmentedColormap.from_list("Ocurrence Map", CMAPlist, CMAP.N)
    # Define Bounds array
    BOUNDS = np.linspace(OcurrenceArray.min(), OcurrenceArray.max(), 7, endpoint=True)
    # Create NORM array
    NORM = BoundaryNorm(BOUNDS, CMAP.N)

    return CMAP, NORM

def Add_TimeMonthsHistogramToPlot(HistogramMonths, CMAP, NORM, Plots, RegionName, StationName, Stat_or_Reg):
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

    HistogramaImagen = Plots[1][0].imshow(HistogramMonths, cmap=CMAP, norm=NORM,
    interpolation="None", aspect="auto", origin="lower", extent=extent)
    colorbar(HistogramaImagen, ax=Plots[1][0], label="% Ocurrence")

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
    usecols=(1, 2), unpack=True, skiprows=1)
    NumMonthTerminator = np.linspace(0.0, 12.0, RiseHours.size)
    Plots[1][0].plot(RiseHours, NumMonthTerminator, "--k", linewidth=1.0)
    Plots[1][0].plot(SetHours, NumMonthTerminator, "--k", linewidth=1.0)

    Plots[0][0].tight_layout()
    if Stat_or_Reg == "Stat":
        SaveStationPlot("OcurrenceTIDs_", RegionName, StationName, Plots[0][0])
    elif Stat_or_Reg == "Reg":
        SaveRegionPlot("OcurrenceTIDs_", RegionName, Plots[0][0])

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Same probability density function as used in lmfit module
# check GaussianModdel in https://lmfit.github.io/lmfit-py/builtin_models.html
GaussianDist = lambda x, A, mu, sigma: (A/(sigma*(2.0*np.pi)**0.5))*np.exp( -0.5*((x-mu)/sigma)**2.0)

def Add_PeriodHistogramToPlot(Period, Time_TIDs, Months_TIDs, Plots, RegionName, StationName, Stat_or_Reg):
    Period = 60.0*Period

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
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
    for PeriodData, Color, NamePlot in zip([NightTIDsPeriods, DayTIDsPeriods], ["blue", "red"], ["Night", "Day"]):

        #Setting number of bins by using the Freedman-Diaconis rule
        Quantiles = mquantiles(PeriodData)
        IQR = Quantiles[2]-Quantiles[0]
        h = 2.0*IQR*(PeriodData.size**(-1/3))
        PeriodRange = (PeriodData.min(), PeriodData.max())
        PeriodBins = int((PeriodRange[1]-PeriodRange[0])/h)

        # Adding density histogram of period data
        PeriodHistogram, Edges, _ = Plots[1][1].hist(PeriodData, bins=PeriodBins, range=PeriodRange, density=True,
                         facecolor=Color, edgecolor="None", alpha=0.5)

        # Stablish the median of each bin as the X value for each density bar
        MidEdges = Edges[:PeriodBins] + np.diff(Edges)

        #Getting mean, deviation of period data and max value of Ocurrence
        Mean, Deviation = PeriodData.mean(), PeriodData.std()
        MaxValue = PeriodHistogram.max()

        #Declaring an Exponential Gaussian Model as the proposed theoretical distribution
        GaussianToFit = GaussianModel()
        #Setting parameters
        ParametersExpGaussian = GaussianToFit.make_params(amplitude=MaxValue,
        center=Mean, sigma=Deviation)
        #Calculate best fit
        ExpGaussianFitResult = GaussianToFit.fit(PeriodHistogram, ParametersExpGaussian, x = MidEdges)

        #Extracting optimal parameters for gaussian fit
        ParamsResults = ExpGaussianFitResult.params
        AmpFit = ParamsResults["amplitude"].value
        MeanFit, MeanError = ParamsResults["center"].value, ParamsResults["center"].stderr
        SigmaFit, SigmaError = ParamsResults["sigma"].value, ParamsResults["sigma"].stderr

        # Create string sequence to show optimal mean and deviation values for the input data
        labelFit = NamePlot+"\n"+r"$\mu$={0:.3f}$\pm${1:.3f}".format(MeanFit,MeanError)+"\n"+r"$\sigma$={0:.3f}$\pm${1:.3f}".format(SigmaFit,SigmaError)

        # Create theoretical distribution given these optimal values
        PeriodLinSampling = np.linspace(PeriodRange[0], 60.0, 200)
        GaussianFitCurve = GaussianDist(PeriodLinSampling, AmpFit, MeanFit, SigmaFit)

        # Adding gaussian curve by using the optimal parameters        
        Plots[1][1].plot(PeriodLinSampling, GaussianFitCurve, linestyle="--", color=Color, linewidth=1.5,
        label=labelFit)

    Plots[1][1].legend()
    Plots[0][1].tight_layout()
    if Stat_or_Reg == "Stat":
        SaveStationPlot("PeriodDistribution_", RegionName, StationName, Plots[0][1])
    elif Stat_or_Reg == "Reg":
        SaveRegionPlot("PeriodDistribution_", RegionName, Plots[0][1])
    
# -------------------------------------------------------------------------------------------------------------------------------------------------
def Add_BarsFreq_Month(Time_TIDs, Months_TIDs, Plots, RegionName, StationName, Stat_or_Reg):
    #Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = np.linspace(0.5,11.5,12,endpoint=True)
    Plots[1][2].set_xticks(MonthAxisData, MonthTicks)

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
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
    if Stat_or_Reg == "Stat":
        SaveStationPlot("DayNightTIDs_", RegionName, StationName, Plots[0][2])
    elif Stat_or_Reg == "Reg":
        SaveRegionPlot("DayNightTIDs_", RegionName, Plots[0][2])

# -------------------------------------------------------------------------------------------------------------------------------------------------
def AddBoxPlot(Plots, Num, Center, Width, dx, InputData, Color):
    if InputData.size  > 0:
        BoxPlot = Plots[1][3][Num].boxplot(InputData, sym="x", positions=[Center + dx], patch_artist=True,
                                            widths=Width)
                
        for ComponentBoxPlot in [BoxPlot["whiskers"], BoxPlot["caps"], BoxPlot["fliers"], BoxPlot["medians"]]:
            for patch in ComponentBoxPlot:
                patch.set_color(Color)
                patch.set_linewidth(2)

        for BoxComponent in BoxPlot["boxes"]:
            BoxComponent.set_facecolor("None")
            BoxComponent.set_edgecolor(Color)

def Add_AmplitudesAnalysis(MinA, MaxA, Time_TIDs, Months_TIDs, Plots, RegionName, StationName, Stat_or_Reg):
    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
    usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:DivH_12], SetHours[0:SizeData:DivH_12]

    # START ANALYSIS GIVEN THE ACTIVITY IN LOCAL TIME
    Indexes = list(range(0,24,2))
    for Index in Indexes:
        MaskTime = np.where((Index <= Time_TIDs) & (Time_TIDs <= Index+2), True, False)
        AverageMinMax_Amps = (np.abs(MinA[MaskTime]) + MaxA[MaskTime])/2.0
        AverageMinMax_Amps = AverageMinMax_Amps.reshape(AverageMinMax_Amps.size, 1)

        AddBoxPlot(Plots, 0, Index, 0.5, 1.0, AverageMinMax_Amps, "k")

    # START ANALYSIS BY MONTHS DIVIDED IN DAY AND NIGHT ACTIVITY
    for month in range(1,13,1):
        Conds_month = Months_TIDs==month
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            MinA_Conds_month = MinA[Conds_month]
            MaxA_Conds_month = MaxA[Conds_month]

            MaskDay = (RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1])
            MaskNight = ~MaskDay

            # Mean and Std of average amplitude from absolute min and max amplitude for...
            # day and ...
            AverageMinMax_DayAmps = (np.abs(MinA_Conds_month[MaskDay]) + MaxA_Conds_month[MaskDay])/2.0

            # night
            AverageMinMax_NightAmps = (np.abs(MinA_Conds_month[MaskNight]) + MaxA_Conds_month[MaskNight])/2.0

            AddBoxPlot(Plots, 1, month, 0.25, 0.25, AverageMinMax_DayAmps, "r")
            AddBoxPlot(Plots, 1, month, 0.25, 0.75, AverageMinMax_NightAmps, "b")

    # Setting x ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = np.linspace(1.0,12.0,12,endpoint=True)
    Plots[1][3][1].set_xticks([])
    Plots[1][3][1].set_xticks(MonthAxisData, MonthTicks)

    # Setting x ticks within 24 hours 
    Plots[1][3][0].set_xticks([])
    XTICKS = [i for i in range(0, 25, 4)]
    Plots[1][3][0].set_xticks(ticks=XTICKS, labels=XTICKS)

    # Setting log scale for y axis in both plots
    for num in range(2):
        Plots[1][3][num].set_yscale("log", subs=None)
       
    Plots[0][3].tight_layout()
    if Stat_or_Reg == "Stat":
        SaveStationPlot("AmpsAnalysis_", RegionName, StationName, Plots[0][3])
    elif Stat_or_Reg == "Reg":
        SaveRegionPlot("AmpsAnalysis_", RegionName, Plots[0][3])

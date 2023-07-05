from matplotlib import use
from matplotlib.pyplot import figure, subplots, colorbar, text
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatterSciNotation

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

IndexName = {
    0: "A) North",
    1: "B) Center",
    2: "C) South"
    }

# Dictionary to extract colors to use in day-night filter for amplitude-power data
DayNightColors = dict(Day="red", Night="blue")

# Logaritmic format to use in Amplitude variance plots
LogFmt = LogFormatterSciNotation(base=10.0, labelOnlyBase=True)

# -------------------------------------------------------------------------------------------------------------------------------------------------

def CreateFiguresForAllRegions(Nplots):
    # ---- CREATE MAIN FIGURE FOR POWER-TIME PLOT ----
    FigurePowerTime = figure(1, figsize=(8, 6))
    SubPlot_PowerTime = FigurePowerTime.add_subplot(111)
    # Add Power-Time labels
    SubPlot_PowerTime.set_xlabel("Local Time (Hours)")
    SubPlot_PowerTime.set_ylabel("TID power")

    # ---- CREATE MAIN FIGURE FOR AMPLITUDE-POWER PLOT ----
    FigureAmplitudePower, AmpPowSub = subplots(num=2, nrows=Nplots, ncols=1, sharex=True, figsize=(6, 6))
    for i in range(Nplots):
        # Add Amplitude-Power labels
        AmpPowSub[i].set_ylabel("TID power")

    AmpPowSub[Nplots-1].set_xlabel("Average absolute amplitude (dTEC)")

    # ---- CREATE MAIN FIGURE FOR AMPLITUDE VARIABILITY PLOT ----
    FigureAmplitudeVar, AmpVarSub = subplots(num=3, nrows=Nplots, ncols=3, sharex="col", figsize=(8, 6))

    # Add Amplitude Variability x-label and y-labels
    for i in range(Nplots):
        AmpVarSub[i][0].set_ylabel("Amplitude (dTEC)")
    AmpVarSub[Nplots-1][0].set_xlabel("Local Time (Hours)")

    # ---- CREATE MAIN FIGURE FOR OCURRENCE PLOT ----
    FigureOcurrHist, OcurrHistSub = subplots(num=4, nrows=Nplots, ncols=1, 
                                             sharex=True, 
                                             figsize=(6, 8))
    # Add Ocurrence x-label
    OcurrHistSub[Nplots-1].set_xlabel("Local Time (Hours)")

    # ---- CREATE MAIN FIGURE FOR PERIOD DISTRIBUTION PLOT ----
    FigurePeriodDists, PeriodDistSub = subplots(num=5, nrows=Nplots, ncols=1, sharex=True, figsize=(6, 6))  
    PeriodDistSub[Nplots-1].set_xlabel("TID Period (Minutes)")
    
    for i in range(Nplots):
        PeriodDistSub[i].set_ylabel("Probability Density")

    # ---- CREATE MAIN FIGURE FOR DAY-NIGHT BARS PLOT ----
    FigureMonthBars, MonthBarsSub = subplots(num=6, nrows=Nplots, ncols=1, sharex=True, figsize=(6, 6))
    # Add Month frequencies labels
    for i in range(Nplots):
        MonthBarsSub[i].set_ylabel("Number of events")            

    return {"POWER_VAR":(FigurePowerTime, SubPlot_PowerTime),
            "AMP_POWER":(FigureAmplitudePower, AmpPowSub),
            "AMP_VAR": (FigureAmplitudeVar, AmpVarSub),
            "OCURR": (FigureOcurrHist, OcurrHistSub),
            "PERIOD":(FigurePeriodDists, PeriodDistSub),
            "DAY-NIGHT_BARS":(FigureMonthBars, MonthBarsSub)
            }

# -------------------------------------------------------------------------------------------------------------------------------------------------


def Add_TimePowerDataResultsToPlot(Time, Power, Plots, Color, Start):
    # Plotting boxplots for each hour interval given the station's index
    Indexes = list(range(Start, 24, 3))
    for Index in Indexes:

        # Creating mask of Time array given a one hour interval
        MaskTime = np.where((Index <= Time) & (Time <= Index+1), True, False)
        PowerMask = Power[MaskTime]
        PowerMask = PowerMask.reshape(PowerMask.size, 1)

        # Create a boxplot only if the size of the Power mask array has elements
        if PowerMask.size > 0:

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
    XTICKS = list(range(0, 25, 4))
    Plots[1].set_xticks(ticks=XTICKS, labels=XTICKS)

    return BoxPlot["boxes"][0]


# -------------------------------------------------------------------------------------------------------------------------------------------------
# Same power law function as used in lmfit module
# check PowerLawModel in https://lmfit.github.io/lmfit-py/builtin_models.html
def PowerFunction(x, A, k): return A*(x**k)


def Add_AmplitudePowerScatterPlot(AverageAmplitude, Power, Time, Months, Plots, Marker, Index, RegionName):
    # Obtain AmplitudeAverage sorting array to sort
    # Power, Time and Months array for future use (filtering and fitting)
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

    print(f"--Power Law Model for Amplitude-Power plot--\nA = {Best_A:.3f}\nExponent = {Best_k:.3f}\nR2-Score = {R2_Score:.3f}\n")

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:DivH_12], SetHours[0:SizeData:DivH_12]

    # Apply day-night filter for amplitude-power data
    DayNightAmplitude = dict(Day=[], Night=[])
    DayNightPower = dict(Day=[], Night=[])

    NumDay, NumNight = 0, 0
    for month in range(1, 13):
        # Separate data by given month
        Conds_month = Months == month

        # Check if there is any data point within this month
        if Conds_month.any():

            # Create arrays given the month and the time to separate in day and night
            Time_Conds_month = Time[Conds_month]
            AveAmplitude_Conds_month = AverageAmplitude[Conds_month]
            Power_Conds_month = Power[Conds_month]

            # Day and night masks and count the total of
            MaskDay = (
                RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1])
            MaskNight = ~MaskDay

            NumDay += MaskDay.sum()
            NumNight += MaskNight.sum()

            DayNightAmplitude["Day"].append(AveAmplitude_Conds_month[MaskDay])
            DayNightAmplitude["Night"].append(
                AveAmplitude_Conds_month[MaskNight])
            DayNightPower["Day"].append(Power_Conds_month[MaskDay])
            DayNightPower["Night"].append(Power_Conds_month[MaskNight])

    DayNightAmplitude["Day"] = np.concatenate(tuple(DayNightAmplitude["Day"]))
    DayNightAmplitude["Night"] = np.concatenate(
        tuple(DayNightAmplitude["Night"]))
    DayNightPower["Day"] = np.concatenate(tuple(DayNightPower["Day"]))
    DayNightPower["Night"] = np.concatenate(tuple(DayNightPower["Night"]))

    # Add scatter plot of average amplitudes and power by Day-Night filter
    for momentDay in ["Night", "Day"]:
        COLOR = DayNightColors[momentDay]
        Plots[1][Index].scatter(
            DayNightAmplitude[momentDay],
            DayNightPower[momentDay],
            alpha=0.25,
            c=COLOR,
            marker=Marker,
            label=f"{RegionName}-{momentDay}",
        )

    # Add best fit of power law model for average amplitudes and power data
    Plots[1][Index].plot(AverageAmplitude, Best_AmpPowerFit, "--k")
    Plots[1][Index].text(0.05, 0.95, f"A = {Best_A:.3f}\nExponent = {Best_k:.3f}\n"+r"$R^{{2}}$ = {0:.3f}".format(R2_Score),
                         horizontalalignment="left", verticalalignment="top", fontsize=9,
                         transform=Plots[1][Index].transAxes)

    Plots[1][Index].set_title(IndexName[Index])

    # And finally, add number of Day and Night events
    Plots[1][Index].text(0.95, 0.05, f"Day = {NumDay} Night = {NumNight}",
                         horizontalalignment="right", verticalalignment="bottom", fontsize=9,
                         transform=Plots[1][Index].transAxes)
    
# -------------------------------------------------------------------------------------------------------------------------------------------------


def AddBoxPlot(Plots, Row, Col, Center, Width, dx, InputData, Color):
    if InputData.size > 0:
        BoxPlot = Plots[1][Row][Col].boxplot(InputData, sym="x", positions=[Center + dx], patch_artist=True,
                                             widths=Width)

        for ComponentBoxPlot in [BoxPlot["whiskers"], BoxPlot["caps"], BoxPlot["fliers"], BoxPlot["medians"]]:
            for patch in ComponentBoxPlot:
                patch.set_color(Color)
                patch.set_linewidth(2)

        for BoxComponent in BoxPlot["boxes"]:
            BoxComponent.set_facecolor("None")
            BoxComponent.set_edgecolor(Color)


def Add_AmplitudesAnalysis(AverageAmplitude, Time_TIDs, Months_TIDs, Plots, Index, RegionName):
    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:
                                    DivH_12], SetHours[0:SizeData:DivH_12]

    # START ANALYSIS GIVEN THE ACTIVITY IN LOCAL TIME
    Hours = list(range(0, 24, 2))
    for Hour in Hours:
        MaskTime = np.where((Hour <= Time_TIDs) & (Time_TIDs <= Hour+2), True, False)
        AverageMinMax_Amps = AverageAmplitude[MaskTime]
        AverageMinMax_Amps = AverageMinMax_Amps.reshape(AverageMinMax_Amps.size, 1)

        AddBoxPlot(Plots, Index, 0, Hour, 0.75, 1.0, AverageMinMax_Amps, "k")

    # START ANALYSIS BY DATE DIVIDED IN DAY AND NIGHT ACTIVITY
    for month in range(1, 13):
        Conds_month = Months_TIDs == month
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            AveAmp_Conds_month = AverageAmplitude[Conds_month]

            # Filter for daytime events
            MaskDay = (RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1])
            AverageMinMax_DayAmps = AveAmp_Conds_month[MaskDay]

            # Filter for nighttime events
            MaskNight = (Time_Conds_month < RiseHours[month-1]) | (SetHours[month-1] <= Time_Conds_month)
            AverageMinMax_NightAmps = AveAmp_Conds_month[MaskNight]

            AddBoxPlot(Plots, Index, 1, month, 0.5, 0.0, AverageMinMax_DayAmps, DayNightColors["Day"])
            AddBoxPlot(Plots, Index, 2, month, 0.5, 0.0, AverageMinMax_NightAmps, DayNightColors["Night"])

    # Apply logaritmic scale to the y-axis in first column
    for n in range(2):
        Plots[1][Index][n].set_yscale("log")
        Plots[1][Index][n].yaxis.set_major_formatter(LogFmt)

    Plots[1][Index][1].set_title(IndexName[Index])

# -------------------------------------------------------------------------------------------------------------------------------------------------
def Add_TimeMonthsHistogramToPlot(HistogramMonths, CMAP, NORM, Plots, Index, Nplots, RegionName):
    TimeRange = (0.0, 24.0)
    MonthRange = (0, 12)
    extent = (*TimeRange, *MonthRange)
    
    HistogramaImagen = Plots[1][Index].imshow(HistogramMonths, cmap=CMAP, norm=NORM, extent=extent,
                                          interpolation="spline36", aspect="auto", origin="lower")
    
    if Index == Nplots - 1:
        ColorBar_Ax = Plots[0].add_axes([0.8, 0.15, 0.05, 0.7])
        colorbar(HistogramaImagen, cax=ColorBar_Ax, label="% Ocurrence")

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    NumMonthTerminator = np.linspace(0.0, 12.0, RiseHours.size)
    Plots[1][Index].plot(RiseHours, NumMonthTerminator, "--k", linewidth=1.0)
    Plots[1][Index].plot(SetHours, NumMonthTerminator, "--k", linewidth=1.0)

    Plots[1][Index].set_title(IndexName[Index])

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Same probability density function as used in lmfit module
# check GaussianModdel in https://lmfit.github.io/lmfit-py/builtin_models.html
def GaussianDist(x, A, mu, sigma): return (
    A/(sigma*(2.0*np.pi)**0.5))*np.exp(-0.5*((x-mu)/sigma)**2.0)

def Add_PeriodHistogramToPlot(Period, Time_TIDs, Months_TIDs, Plots, Index, RegionName):
    Period = 60.0*Period

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:DivH_12], SetHours[0:SizeData:DivH_12]

    DayTIDsPeriods = []
    NightTIDsPeriods = []
    for month in range(1, 13):
        Conds_month = Months_TIDs == month
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            Period_Conds_month = Period[Conds_month]
            MaskDay = (RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1])
            MaskNight = ~MaskDay

            DayTIDsPeriods.append(Period_Conds_month[MaskDay])
            NightTIDsPeriods.append(Period_Conds_month[MaskNight])

    DayTIDsPeriods = np.concatenate(tuple(DayTIDsPeriods))
    NightTIDsPeriods = np.concatenate(tuple(NightTIDsPeriods))
    for PeriodData, NamePlot in zip([NightTIDsPeriods, DayTIDsPeriods], ["Night", "Day"]):

        # Setting number of bins by using the Freedman-Diaconis rule
        Quantiles = mquantiles(PeriodData)
        IQR = Quantiles[2]-Quantiles[0]
        h = 2.0*IQR*(PeriodData.size**(-1/3))
        PeriodRange = (PeriodData.min(), PeriodData.max())
        PeriodBins = int((PeriodRange[1]-PeriodRange[0])/h)

        # Extract color
        Color = DayNightColors[NamePlot]

        # Adding density histogram of period data
        PeriodHistogram, Edges, _ = Plots[1][Index].hist(PeriodData, bins=PeriodBins, range=PeriodRange, density=True,
                                                     facecolor=Color, edgecolor="None", alpha=0.5)

        # Stablish the median of each bin as the X value for each density bar
        MidEdges = Edges[:PeriodBins] + np.diff(Edges)

        # Getting mean, deviation of period data and max value of Ocurrence
        Mean, Deviation = PeriodData.mean(), PeriodData.std()
        MaxValue = PeriodHistogram.max()

        # Declaring an Exponential Gaussian Model as the proposed theoretical distribution
        GaussianToFit = GaussianModel()
        # Setting parameters
        ParametersExpGaussian = GaussianToFit.make_params(amplitude=MaxValue,
                                                          center=Mean, sigma=Deviation)
        # Calculate best fit
        ExpGaussianFitResult = GaussianToFit.fit(
            PeriodHistogram, ParametersExpGaussian, x=MidEdges)

        # Extracting optimal parameters for gaussian fit
        ParamsResults = ExpGaussianFitResult.params
        AmpFit = ParamsResults["amplitude"].value
        MeanFit, MeanError = ParamsResults["center"].value, ParamsResults["center"].stderr
        SigmaFit, SigmaError = ParamsResults["sigma"].value, ParamsResults["sigma"].stderr

        # Create string sequence to show optimal mean and deviation values for the input data
        labelFit = NamePlot+"\n"+r"$\mu$={0:.3f}$\pm${1:.3f}".format(
            MeanFit, MeanError)+"\n"+r"$\sigma$={0:.3f}$\pm${1:.3f}".format(SigmaFit, SigmaError)

        # Create theoretical distribution given these optimal values
        PeriodLinSampling = np.linspace(PeriodRange[0], 60.0, 200)
        GaussianFitCurve = GaussianDist(
            PeriodLinSampling, AmpFit, MeanFit, SigmaFit)

        # Adding gaussian curve by using the optimal parameters
        Plots[1][Index].plot(PeriodLinSampling, GaussianFitCurve, linestyle="--", color=Color, linewidth=1.5,
                         label=labelFit)

    Plots[1][Index].set_title(IndexName[Index])
    Plots[1][Index].legend()

# -------------------------------------------------------------------------------------------------------------------------------------------------


def Add_BarsFreq_Month(Time_TIDs, Months_TIDs, Plots, Index, Nplots, RegionName):
    # Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May",
                  "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = np.linspace(0.5, 11.5, 12, endpoint=True)
    Plots[1][Index].set_xticks(MonthAxisData, MonthTicks)

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:DivH_12], SetHours[0:SizeData:DivH_12]

    # Count number of events per month given the rise and set hours of the sun
    NumEventerPerMonth = []
    for month in range(1, 13):
        Conds_month = Months_TIDs == month
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            NumDayNight_month = np.where(
                (RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1]), 1, 0)
            NumDay = (NumDayNight_month == 1).sum()
            NumNight = (NumDayNight_month == 0).sum()

            NumEventerPerMonth.append((month, NumDay, NumNight))
        else:
            NumEventerPerMonth.append((month, 0, 0))

    # Sort the array given the month number
    NumEventerPerMonth.sort(key=lambda e: e[0])
    # Unzip the lists for number of events in day and night in this Station/Region
    _, NumEvents_Day, NumEvents_Night = zip(*NumEventerPerMonth)

    DayBars = Plots[1][Index].bar(x=MonthAxisData - 0.25, height=NumEvents_Day, width=0.25,
                        align="edge", edgecolor="k", facecolor="r")
    NightBars = Plots[1][Index].bar(x=MonthAxisData, height=NumEvents_Night, width=0.25,
                        align="edge", edgecolor="k", facecolor="b")

    if Index == Nplots - 1:
        # Fixing position of legends box outside the subplots after FormatPlots.py is called
        Plots[0].legend([DayBars, NightBars], ["Day", "Night"], 
                                          loc="center right", bbox_to_anchor=(1, 0.5),
                                          fancybox=True, shadow=True)

    Plots[1][Index].set_title(IndexName[Index])

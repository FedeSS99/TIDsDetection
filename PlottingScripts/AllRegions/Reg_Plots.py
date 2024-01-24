from matplotlib.pyplot import subplots, colorbar
from matplotlib.ticker import LogFormatterSciNotation

from scipy.stats import iqr, skew
from scipy.special import erf
from scipy.stats.mstats import mquantiles
import numpy as np
from lmfit.models import SkewedGaussianModel, PowerLawModel

from json import load

from DataScripts.CommonDictionaries import TerminatorsDict, DayNightColors, IndexName

# Logaritmic format to use in amplitude and power variance plots
LogFmt = LogFormatterSciNotation(base=10.0, labelOnlyBase=True)

# -------------------------------------------------------------------------------------------------------------------------------------------------
def CreateFiguresForAllRegions(Nplots):
    # ---- CREATE MAIN FIGURE FOR POWER VARIABILITY ----
    FigurePowerVar, PowerVarSub = subplots(
        num=1, nrows=Nplots, ncols=3, sharex="col", figsize=(8, 6))

    # Add Power Variability x-label and y-labels
    PowerVarWindSpeedSub = []
    for i in range(Nplots):
        PowerVarSub[i][0].set_ylabel("TID Power (dTEC²)", fontsize=8)
        PowerVarSub[i][1].set_ylabel("IQR-Power (dTEC²)", fontsize=8)
        PowerVarWindSpeedSub.append(
            [PowerVarSub[i][n].twinx() for n in range(1, 3)])

        PowerVarWindSpeedSub[i][1].set_ylabel("Wind speed (m/s)", fontsize=8)
    PowerVarSub[Nplots-1][0].set_xlabel("Local Time (Hours)")

    # ---- CREATE MAIN FIGURE FOR AMPLITUDE-POWER ----
    FigureAmplitudePower, AmpPowSub = subplots(
        num=2, nrows=Nplots, ncols=1, sharex=True, figsize=(6, 6))
    for i in range(Nplots):
        # Add Amplitude-Power labels
        AmpPowSub[i].set_ylabel("TID power")

    AmpPowSub[Nplots-1].set_xlabel("Average absolute amplitude (dTEC)")

    # ---- CREATE MAIN FIGURE FOR AMPLITUDE VARIABILITY ----
    FigureAmplitudeVar, AmpVarSub = subplots(
        num=3, nrows=Nplots, ncols=3, sharex="col", figsize=(8, 6))

    # Add Amplitude Variability x-label and y-labels
    AmpVarWindSpeedSub = []
    for i in range(Nplots):
        AmpVarSub[i][0].set_ylabel("Amplitude (dTEC)", fontsize=8)
        AmpVarSub[i][1].set_ylabel("IQR-AMA (dTEC)", fontsize=8)
        AmpVarWindSpeedSub.append([AmpVarSub[i][n].twinx()
                                  for n in range(1, 3)])

        AmpVarWindSpeedSub[i][1].set_ylabel("Wind speed (m/s)", fontsize=8)
    AmpVarSub[Nplots-1][0].set_xlabel("Local Time (Hours)")

    # ---- CREATE MAIN FIGURE FOR OCURRENCE ----
    FigureOcurrHist, OcurrHistSub = subplots(
        num=4, nrows=Nplots, ncols=1, sharex=True, figsize=(6, 8))
    # Add Ocurrence x-label
    OcurrHistSub[Nplots-1].set_xlabel("Local Time (Hours)")

    # ---- CREATE MAIN FIGURE FOR PERIOD DISTRIBUTION ----
    FigurePeriodDists, PeriodDistSub = subplots(
        num=5, nrows=Nplots, ncols=2, sharex="all", sharey="all", figsize=(8, 6))
    PeriodDistSub[Nplots-1][0].set_xlabel("TID Period (Minutes)")
    PeriodDistSub[Nplots-1][1].set_xlabel("TID Period (Minutes)")

    for i in range(Nplots):
        PeriodDistSub[i][0].set_ylabel("Prob. Density")

    # ---- CREATE MAIN FIGURE FOR DAY-NIGHT BARS ----
    FigureMonthBars, MonthBarsSub = subplots(
        num=6, nrows=Nplots, ncols=1, sharex=True, figsize=(6, 6))
    # Add Month frequencies labels
    for i in range(Nplots):
        MonthBarsSub[i].set_ylabel("Number of events")

    # ---- CREATE MAIN FIGURE FOR POWER TIME DISTRIBUTION ----
    FigurePowerHist, PowerHistSub = subplots(
        num=7, nrows=Nplots, ncols=1, sharex=True, figsize=(6, 8))
    # Add Ocurrence x-label
    PowerHistSub[Nplots-1].set_xlabel("Local Time (Hours)")

    return {"POWER-VAR": (FigurePowerVar, PowerVarSub, PowerVarWindSpeedSub),
            "AMP-POWER": (FigureAmplitudePower, AmpPowSub),
            "AMP-VAR": (FigureAmplitudeVar, AmpVarSub, AmpVarWindSpeedSub),
            "OCURR": (FigureOcurrHist, OcurrHistSub),
            "PERIOD": (FigurePeriodDists, PeriodDistSub),
            "DAY-NIGHT-BARS": (FigureMonthBars, MonthBarsSub),
            "POWER-MED": (FigurePowerHist, PowerHistSub)
            }

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

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:
                                    DivH_12], SetHours[0:SizeData:DivH_12]

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
            label=f"{momentDay}\n{RegionName}",
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


def AddBarPlot(Plots, Row, Col, BarCenter, Height, Width, Color):
    Plots[1][Row][Col].bar(BarCenter, Height, width=Width,
                           color=Color, edgecolor="black")


def Add_QuantityVarAnalysis(Quantity, Time_TIDs, Months_TIDs, Plots, Index, RegionName):
    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:
                                    DivH_12], SetHours[0:SizeData:DivH_12]

    # START ANALYSIS GIVEN THE ACTIVITY IN LOCAL TIME
    Hours = np.array(list(range(0, 24, 2)))
    LowerQ = len(Hours)*[0.0]
    Median = len(Hours)*[0.0]
    HigherQ = len(Hours)*[0.0]
    for n, Hour in enumerate(Hours):
        MaskTime = np.where((Hour <= Time_TIDs) & (
            Time_TIDs <= Hour+2), True, False)
        AverageQuantity = Quantity[MaskTime]
        TwoHourQuantityQuantiles = mquantiles(AverageQuantity)

        LowerQ[n] = TwoHourQuantityQuantiles[0]
        Median[n] = TwoHourQuantityQuantiles[1]
        HigherQ[n] = TwoHourQuantityQuantiles[2]

    Plots[1][Index][0].fill_between(Hours+1, HigherQ, LowerQ, alpha=0.5, linewidth=0,
                                    color="black")
    Plots[1][Index][0].plot(Hours+1, Median, color="black", linewidth=1)

    # START ANALYSIS BY DATE DIVIDED IN DAY AND NIGHT ACTIVITY

    for month in range(1, 13):
        Conds_month = (Months_TIDs == month)
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            Quantity_Conds_month = Quantity[Conds_month]

            # Filter for daytime events
            MaskDay = (
                RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1])
            Quantity_Day = Quantity_Conds_month[MaskDay]
            DispersionDay = iqr(Quantity_Day)

            # Filter for nighttime events
            MaskNight = (
                Time_Conds_month < RiseHours[month-1]) | (SetHours[month-1] < Time_Conds_month)
            Quantity_Night = Quantity_Conds_month[MaskNight]
            DispersionNight = iqr(Quantity_Night)

            AddBarPlot(Plots, Index, 1, month+0.5, DispersionDay, 0.5, "red")
            AddBarPlot(Plots, Index, 2, month+0.5,
                       DispersionNight, 0.5, "blue")

    # Extract wind dispersion data0
    with open("./WindSpeedData/VelocityMonthDispersion.json", "r") as WS_JSON:
        WindSpeedData = load(WS_JSON)

    Width = 0.125
    dx = 0.5
    MonthsDay = [x + dx - Width for x in range(1, 13)]
    MonthsNight = [x + dx + Width for x in range(1, 13)]
    for Region in WindSpeedData.keys():
        if RegionName in Region:

            WindSpeedData[Region] = np.array(WindSpeedData[Region])

            Plots[2][Index][0].fill_between(MonthsDay, WindSpeedData[Region][0, :], WindSpeedData[Region][2, :],
                                            color="red", alpha=0.25)
            Plots[2][Index][0].plot(
                MonthsDay, WindSpeedData[Region][1, :], "--r", linewidth=1)
            Plots[2][Index][1].fill_between(MonthsNight, WindSpeedData[Region][3, :], WindSpeedData[Region][5, :],
                                            color="blue", alpha=0.25)
            Plots[2][Index][1].plot(
                MonthsNight, WindSpeedData[Region][4, :], "--b", linewidth=1)

    Plots[1][Index][1].set_title(IndexName[Index])

# -------------------------------------------------------------------------------------------------------------------------------------------------


def Add_TimeMonthsHistogramToPlot(HistogramMonths, CMAP, NORM, Plots, Index, Nplots, RegionName, Label):
    TimeRange = (0.0, 24.0)
    MonthRange = (0, 12)
    extent = (*TimeRange, *MonthRange)

    HistogramaImagen = Plots[1][Index].imshow(HistogramMonths, cmap=CMAP, norm=NORM, extent=extent,
                                              interpolation="spline36", aspect="auto", origin="lower")

    if Index == Nplots - 1:
        ColorBar_Ax = Plots[0].add_axes([0.8, 0.15, 0.05, 0.7])
        colorbar(HistogramaImagen, cax=ColorBar_Ax, label=Label)

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    NumMonthTerminator = np.linspace(0.0, 12.0, RiseHours.size)
    Plots[1][Index].plot(RiseHours, NumMonthTerminator, "--k", linewidth=1.0)
    Plots[1][Index].plot(SetHours, NumMonthTerminator, "--k", linewidth=1.0)

    Plots[1][Index].set_title(IndexName[Index])

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Same probability density function as used in lmfit module
# check GaussianModel in https://lmfit.github.io/lmfit-py/builtin_models.html


def SkewedGaussianDist(x, A, mu, sigma, gamma): 
    return (A/(sigma*(2.0*np.pi)**0.5))*np.exp(-0.5*((x-mu)/sigma)**2.0)*(1 + erf(gamma*(x - mu)/(sigma*(2**0.5))))


def Add_PeriodHistogramToPlot(Period, Time_TIDs, Months_TIDs, Plots, Index, RegionName):
    Period = 60.0*Period

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:
                                    DivH_12], SetHours[0:SizeData:DivH_12]

    DayTIDsPeriods = []
    NightTIDsPeriods = []
    for month in range(1, 13):
        Conds_month = Months_TIDs == month
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            Period_Conds_month = Period[Conds_month]
            MaskDay = (
                RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1])
            MaskNight = ~MaskDay

            DayTIDsPeriods.append(Period_Conds_month[MaskDay])
            NightTIDsPeriods.append(Period_Conds_month[MaskNight])

    DayTIDsPeriods = np.concatenate(tuple(DayTIDsPeriods))
    NightTIDsPeriods = np.concatenate(tuple(NightTIDsPeriods))
    for n, Periods_Name in enumerate(zip([DayTIDsPeriods, NightTIDsPeriods], ["Day", "Night"])):
        # Setting number of bins by using the Freedman-Diaconis rule
        #IQR = iqr(Periods_Name[0])
        #h = 2.0*IQR*(Periods_Name[0].size**(-1/3))
        PeriodRange = (Periods_Name[0].min(), Periods_Name[0].max())
        #PeriodBins = int((PeriodRange[1]-PeriodRange[0])/h)
        #PeriodBins = int(np.ceil(np.log2(Periods_Name[0].size)) + 1)
        size = Periods_Name[0].size
        PeriodBins = 1 + int(np.log2(size) + np.log2(1 + abs(skew(Periods_Name[0]))/np.sqrt(6*(size-2)/((size+1)*(size+3)))))

        # Extract color
        Color = DayNightColors[Periods_Name[1]]

        # Adding density histogram of period data
        PeriodHistogram, Edges, _ = Plots[1][Index][n].hist(Periods_Name[0], bins=PeriodBins, range=PeriodRange, density=True,
                                                         facecolor=Color, edgecolor="None", alpha=0.5)

        # Stablish the median of each bin as the X value for each density bar
        MidEdges = Edges[:PeriodBins] + 0.5*np.diff(Edges)

        # Getting mean, deviation and skewness of period data and max value of Ocurrence
        Mean, Deviation = Periods_Name[0].mean(), Periods_Name[0].std()
        MaxValue = PeriodHistogram.max()
        Skewness = skew(Periods_Name[0])

        # Declaring an Exponential Gaussian Model as the proposed theoretical distribution
        SkewGaussianToFit = SkewedGaussianModel()
        # Setting parameters
        ParametersExpGaussian = SkewGaussianToFit.make_params(amplitude=MaxValue, center=Mean, 
                                                          sigma=Deviation, gamma=Skewness)
        # Calculate best fit
        SkewGaussianFitResult = SkewGaussianToFit.fit(PeriodHistogram, ParametersExpGaussian, x=MidEdges)

        # Extracting optimal parameters for gaussian fit
        ParamsResults = SkewGaussianFitResult.params
        AmpFit = ParamsResults["amplitude"].value
        MeanFit, MeanError = ParamsResults["center"].value, ParamsResults["center"].stderr
        SigmaFit, SigmaError = ParamsResults["sigma"].value, ParamsResults["sigma"].stderr
        SkewFit, SkewError = ParamsResults["gamma"].value, ParamsResults["gamma"].stderr

        # Create string sequence to show optimal mean and deviation values for the input data
        labelFit = Periods_Name[1]+"\n"+r"$\mu$={0:.3f}$\pm${1:.3f}".format(MeanFit, MeanError)
        labelFit += "\n"
        labelFit += r"$\sigma$={0:.3f}$\pm${1:.3f}".format(SigmaFit, SigmaError)
        labelFit += "\n"
        labelFit += r"$\gamma$={0:.3f}$\pm${1:.3f}".format(SkewFit, SkewError)

        # Create theoretical distribution given the optimal values
        PeriodLinSampling = np.linspace(PeriodRange[0], 60.0, 200, endpoint=True)
        GaussianFitCurve = SkewedGaussianDist(PeriodLinSampling, AmpFit, MeanFit, SigmaFit, SkewFit)

        # Adding gaussian curve by using the optimal parameters
        Plots[1][Index][n].plot(PeriodLinSampling, GaussianFitCurve, linestyle="--", color=Color, linewidth=1.5,
                             label=labelFit)

        Plots[1][Index][n].legend(prop={"size": 8})


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
            NumDayNight_month = np.where((RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1]), 1, 0)
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

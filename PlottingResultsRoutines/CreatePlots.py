from statistics import quantiles
from matplotlib import use
from matplotlib.pyplot import figure, rcParams, colorbar

from numpy import pi, arange, linspace, where, histogram, float64, loadtxt, diff, exp
from lmfit.models import GaussianModel

use("TkAgg")

def CreateResultsFigurePower():
    #Create main figure for each analysis
    FigurePowerTime = figure(1, figsize=(8,6))
    SubPlot_PowerTime = FigurePowerTime.add_subplot(111)

    #Setting titles
    SubPlot_PowerTime.set_title(f"Temporal distribution of TIDs' power")

    #Power-Time labels and background color
    SubPlot_PowerTime.set_xlabel("Local Time (Hours)")
    SubPlot_PowerTime.set_ylabel("TID power")

    #Setting time ticks
    timeTicks = arange(0, 25, 6)
    SubPlot_PowerTime.set_xlim(0.0, 24.0)
    SubPlot_PowerTime.set_xticks(timeTicks)

    return (FigurePowerTime, SubPlot_PowerTime)


def CreateFiguresResults(Coord):
    #Create main figure for each analysis
    FigureTimeHist = figure(2, figsize=(6,6))
    FigurePeriodsHist = figure(3, figsize=(6,6))
    Figure_MonthBars = figure(4, figsize=(6,6))

    SubPlot_TimeHist = FigureTimeHist.add_subplot(111)
    SubPlot_PeriodHist = FigurePeriodsHist.add_subplot(111)
    SubPlot_MonthsFreq = Figure_MonthBars.add_subplot(111)

    #Setting titles
    Coord_String = f"\nLocation: {Coord}"
    SubPlot_TimeHist.set_title(f"Ocurrence of TIDs through seasons and LT"+Coord_String)
    SubPlot_PeriodHist.set_title(f"Distribution of observed periods"+Coord_String)
    SubPlot_MonthsFreq.set_title(f"Number of TIDs by Day-Night"+Coord_String)

    #Time Histogram labels
    SubPlot_TimeHist.set_xlabel("Local Time (Hours)")
    SubPlot_TimeHist.set_ylabel("Month")
    #Period Histogram labels
    SubPlot_PeriodHist.set_xlabel("TID Period (Minutes)")
    SubPlot_PeriodHist.set_ylabel("% Ocurrence")
    #Month frequencies tables
    SubPlot_MonthsFreq.set_ylabel("Number of events")

    return [(FigureTimeHist, FigurePeriodsHist, Figure_MonthBars), (SubPlot_TimeHist, SubPlot_PeriodHist, SubPlot_MonthsFreq)]

def addTimePowerDataResultsToPlot(Time, Power, Plots, Color, Name, index):
    #Plotting results
    if index == 0:
        Plots[1].plot(Time, Power, "o", color=Color, markersize=2.0, alpha=0.25)
    else:
        #Lists to save middle times, mean power and deviation
        MidTimes = []
        MeanPerHours = []
        for i in range(24):
            MaskTime = where((i <= Time) & (Time <= i+1), True, False)
            PowerMask = Power[MaskTime]
            if PowerMask.size  > 0:
                MeanPower = PowerMask.mean()

                MidTimes.append(i + 0.5)
                MeanPerHours.append(MeanPower)
        Plots[1].plot(MidTimes, MeanPerHours, "-", color=Color, linewidth=2, markersize=2,label=Name)


def AddTimeMonthsHistogramToPlot(HistogramMonths, absMin, absMax, Plots, Name):
    #Setting number of bins and time range for histogram
    TimeBins = 24
    TimeRange = (0.0, 24.0)
    MonthBins = 12
    MonthRange = (0, 12)
    #Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = linspace(0.5,11.5,12,endpoint=True)
    Plots[1][0].set_yticks(MonthAxisData, MonthTicks)

    extent = (*TimeRange,*MonthRange)
    timeTicks = arange(0, 25, 6)
    Plots[1][0].set_xlim(*TimeRange)
    Plots[1][0].set_ylim(*MonthRange)
    Plots[1][0].set_xticks(timeTicks)

    #Extract terminator hours
    if "NorthEasth" in Name:
        TerminatorsFile = "TerminatorHours_NE.dat"
    elif "SouthWest" in Name:
        TerminatorsFile = "TerminatorHours_SW.dat"
    elif "NorthWest" in Name:
        TerminatorsFile = "TerminatorHours_NW.dat"
    elif "SouthEast" in Name:
        TerminatorsFile = "TerminatorHours_SE.dat"

    RiseHours, SetHours = loadtxt(TerminatorsFile, dtype=float64,
    usecols=(1, 2), unpack=True, skiprows=1)
    NumMonthTerminator = linspace(0.0, 12.0, RiseHours.size)

    HistogramaImagen = Plots[1][0].imshow(HistogramMonths, cmap="cubehelix", interpolation="bessel",
    vmin=absMin, vmax=absMax, aspect="auto", origin="lower", extent=extent)
    colorbar(HistogramaImagen, ax=Plots[1][0], label="% Ocurrence")
    Plots[1][0].plot(RiseHours, NumMonthTerminator, "--w", linewidth=1.0)
    Plots[1][0].plot(SetHours, NumMonthTerminator, "--w", linewidth=1.0)

    Plots[0][0].savefig(f"./../Resultados/{Name}/OcurrenceTIDs_{Name}.png")


Gaussian_Dist = lambda x, A, sig, mu: (A/(sig*(2.0*pi)**0.5))*exp(-0.5*((x-mu)/sig)**2.0)
def AddPeriodHistogramToPlot(Period, Plots, Name):
    #Setting number of bins, period range and also the width of the bars
    Period = 60.0*Period
    Quantiles = quantiles(Period, n=4)
    h = 2.0*(Quantiles[2]-Quantiles[0])*(Period.size**(-1/3))
    PeriodRange = (Period.min(), Period.max())
    PeriodBins = int((PeriodRange[1]-PeriodRange[0])/h)
    BarsPeriod = linspace(PeriodRange[0], PeriodRange[1], PeriodBins)
    Width = diff(BarsPeriod).mean()

    #Obtaining histogram from period data with given bins and range
    PeriodHistogram, Edges = histogram(Period, bins=PeriodBins, range=PeriodRange)
    #Calculate percentage for each bin
    Ocurrence = 100.0*PeriodHistogram/PeriodHistogram.sum()

    #Getting mean, deviation of period data and max value of Ocurrence
    Mean, Deviation = Period.mean(), Period.std()
    MaxOcurrence = Ocurrence.max()

    #Declaring an Exponential Gaussian Model as the proposed theorical distribution
    GaussianToFit = GaussianModel()
    #Setting parameters
    ParametersExpGaussian = GaussianToFit.make_params(amplitude=MaxOcurrence,
    center=Mean, sigma=Deviation)
    #Calculate best fit
    ExpGaussianFitResult = GaussianToFit.fit(Ocurrence, ParametersExpGaussian, x=Edges[1:])

    ParamsResults = ExpGaussianFitResult.params
    AmpFit = ParamsResults["amplitude"].value
    MeanFit, MeanError = ParamsResults["center"].value, ParamsResults["center"].stderr
    SigmaFit, SigmaError = ParamsResults["sigma"].value, ParamsResults["sigma"].stderr

    PRange = linspace(0.0, max(PeriodRange), 100)
    labelFit = r"$\mu$={0:.3f}$\pm${1:.3f}".format(MeanFit,MeanError)+"\n"+r"$\sigma$={0:.3f}$\pm${1:.3f}".format(SigmaFit,SigmaError)

    Plots[1][1].set_xlim(0.0, max(PeriodRange))
    Plots[1][1].bar(BarsPeriod, height=Ocurrence, width=Width, align="edge",
    facecolor="b", edgecolor="k")
    Plots[1][1].plot(PRange, Gaussian_Dist(PRange,AmpFit,SigmaFit,MeanFit), "r--", linewidth=1.5,
    label=labelFit)

    Plots[1][1].legend()
    Plots[0][1].savefig(f"./../Resultados/{Name}/PeriodDistribution_{Name}.png")


def BarsFreq_Month(Time_TIDs, Months_TIDs, Plots, Name):
    #Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = linspace(0.5,11.5,12,endpoint=True)
    Plots[1][2].set_xticks(MonthAxisData, MonthTicks)

    #Extract terminator hours
    if "NorthEasth" in Name:
        TerminatorsFile = "TerminatorHours_NE.dat"
    elif "SouthWest" in Name:
        TerminatorsFile = "TerminatorHours_SW.dat"
    elif "NorthWest" in Name:
        TerminatorsFile = "TerminatorHours_NW.dat"
    elif "SouthEast" in Name:
        TerminatorsFile = "TerminatorHours_SE.dat"

    RiseHours, SetHours = loadtxt(TerminatorsFile, dtype=float64,
    usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:DivH_12], SetHours[0:SizeData:DivH_12]

    NumEventerPerMonth = []
    Months = [i for i in range(1, 13)]
    for month in Months:
        Conds_month = Months_TIDs==month
        if Conds_month.any():
            Time_Conds_month = Time_TIDs[Conds_month]
            NumDayNight_month = where((RiseHours[month-1] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month-1]), 1, 0)
            NumDay = (NumDayNight_month==1).sum()
            NumNight = (NumDayNight_month==0).sum()

            NumEventerPerMonth.append((month, NumDay, NumNight))
        else:
            NumEventerPerMonth.append((month, 0, 0))

    NumEventerPerMonth.sort(key = lambda e: e[0])
    _, NumEvents_Day, NumEvents_Night = zip(*NumEventerPerMonth)

    #Plots[1][3].bar(x=MonthAxisData[:NumBars] - 0.5, height=NumEvents_Day, width=1,
    #align="edge", edgecolor="k", facecolor="r", label="Day")
    #Plots[1][3].bar(x=MonthAxisData[:NumBars] - 0.5, height=NumEvents_Night, width=1,
    #align="edge", edgecolor="k", facecolor="b", label="Night", bottom=NumEvents_Day)

    Plots[1][2].legend()
    Plots[0][2].savefig(f"./../Resultados/{Name}/DayNightTIDs_{Name}.png")

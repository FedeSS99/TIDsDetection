from matplotlib import use
from matplotlib.pyplot import figure, colorbar

from statistics import quantiles
import numpy as np
from scipy.interpolate import interp1d
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
    Coord_String = f"\n{Coord} Region"
    SubPlot_TimeHist.set_title(f"Annual Ocurrence of TIDs through LT"+Coord_String)
    SubPlot_PeriodHist.set_title(f"Distribution of observed periods"+Coord_String)
    SubPlot_MonthsFreq.set_title(f"Number of TIDs by Day-Night"+Coord_String)

    #Time Histogram labels
    SubPlot_TimeHist.set_xlabel("Local Time (Hours)")
    SubPlot_TimeHist.set_ylabel("Month")
    #Period Histogram labels
    SubPlot_PeriodHist.set_xlabel("TID Period (Minutes)")
    SubPlot_PeriodHist.set_ylabel("Probability Density")
    #Month frequencies tables
    SubPlot_MonthsFreq.set_ylabel("Number of events")

    return [(FigureTimeHist, FigurePeriodsHist, Figure_MonthBars), (SubPlot_TimeHist, SubPlot_PeriodHist, SubPlot_MonthsFreq)]

def SavePlot(GenName, RegName, PlotFigure):
    for format in ["png", "pdf"]:
        PlotFigure.savefig(f"./../Resultados/{RegName}/{GenName}_{RegName}.{format}")

def addTimePowerDataResultsToPlot(Time, Power, Plots, Color, Start):
    #Plotting mean power with its deviation
    #Lists to save middle times, mean power and deviation
    #MidTimes = []
    #MeanPerHours = []
    #StdPerHours = []
    Indexes = list(range(Start, 24, 3))
    LastIndex = Indexes[-1]
    DictBoxPlots = dict()
    ClearLabels = len(Indexes)*[]
    for Index in Indexes:
        MaskTime = np.where((Index <= Time) & (Time <= Index+1), True, False)
        PowerMask = Power[MaskTime]
        PowerMask = PowerMask.reshape(PowerMask.size, 1)
        if PowerMask.size  > 0:
            #MeanPower = PowerMask.mean()
            #StdPower = PowerMask.std()

            #MidTimes.append(i + 0.5)
            #MeanPerHours.append(MeanPower)
            #StdPerHours.append(StdPower)

            BoxPlot = Plots[1].boxplot(PowerMask, sym="x", positions=[Index + 0.5], patch_artist=True,
                                        widths=0.25)
            
            for ComponentBoxPlot in [BoxPlot["whiskers"], BoxPlot["caps"], BoxPlot["fliers"], BoxPlot["medians"]]:
                for patch in ComponentBoxPlot:
                    patch.set_color(Color)
                    patch.set_linewidth(2)

            for BoxComponent in BoxPlot["boxes"]:
                BoxComponent.set_facecolor("None")
                BoxComponent.set_edgecolor(Color)

    #Plots[1].errorbar(x=MidTimes, y=MeanPerHours, yerr=StdPerHours, ecolor=Color,
    #                color=Color, elinewidth=2.0, capthick=2.0, capsize=10.0, fmt="o",
    #                label=Name)

    Plots[1].set_xticks([])
    XTICKS = [i for i in range(0, 25, 4)]
    Plots[1].set_xticks(ticks=XTICKS, labels=XTICKS)

    return BoxPlot["boxes"][0]


def AddTimeMonthsHistogramToPlot(HistogramMonths, absMin, absMax, Plots, Name):
    #Setting number of bins and time range for histogram
    TimeRange = (0.0, 24.0)
    MonthRange = (0, 12)
    #Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = np.linspace(0.5,11.5,12,endpoint=True)
    Plots[1][0].set_yticks(MonthAxisData, MonthTicks)

    extent = (*TimeRange,*MonthRange)
    timeTicks = np.arange(0, 25, 6)
    Plots[1][0].set_xlim(*TimeRange)
    Plots[1][0].set_ylim(*MonthRange)
    Plots[1][0].set_xticks(timeTicks)

    #ImageArray = np.ma.masked_where(HistogramMonths == 0.0, HistogramMonths)
    #HistogramaImagen = Plots[1][0].imshow(ImageArray, cmap="nipy_spectral",
    #vmin=absMin, vmax=absMax, aspect="auto", origin="lower", extent=extent)
    HistogramaImagen = Plots[1][0].imshow(HistogramMonths, cmap="nipy_spectral", interpolation="none",
    vmin=absMin, vmax=absMax, aspect="auto", origin="lower", extent=extent)
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
    Plots[1][0].plot(RiseHours, NumMonthTerminator, "--w", linewidth=1.0)
    Plots[1][0].plot(SetHours, NumMonthTerminator, "--w", linewidth=1.0)

    SavePlot("OcurrenceTIDs", Name, Plots[0][0])


Gaussian_Dist = lambda x, A, sig, mu: (A/(sig*(2.0*np.pi)**0.5))*np.exp(-0.5*((x-mu)/sig)**2.0)
def AddPeriodHistogramToPlot(Period, Time_TIDs, Months_TIDs, Plots, Name):
    
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
    for PeriodData, Color, NamePlot in zip([DayTIDsPeriods, NightTIDsPeriods], ["red", "blue"], ["Day", "Night"]):
        #Setting number of bins, period range and also the width of the bars
        Quantiles = quantiles(PeriodData, n=4)
        h = 2.0*(Quantiles[2]-Quantiles[0])*(PeriodData.size**(-1/3))
        PeriodRange = (PeriodData.min(), PeriodData.max())
        PeriodBins = int((PeriodRange[1]-PeriodRange[0])/h)
        BarsPeriod = np.linspace(PeriodRange[0], PeriodRange[1], PeriodBins)
        Width = np.diff(BarsPeriod).mean()

        #Obtaining histogram from period data with given bins and range
        PeriodHistogram, Edges = np.histogram(PeriodData, bins=PeriodBins, range=PeriodRange)
        #Calculate percentage for each bin
        Ocurrence = 100.0*PeriodHistogram/PeriodHistogram.sum()

        #Getting mean, deviation of period data and max value of Ocurrence
        Mean, Deviation = PeriodData.mean(), PeriodData.std()
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

        PeriodRange = np.linspace(0.0, max(PeriodRange), 100)
        labelFit = NamePlot+"\n"+r"$\mu$={0:.3f}$\pm${1:.3f}".format(MeanFit,MeanError)+"\n"+r"$\sigma$={0:.3f}$\pm${1:.3f}".format(SigmaFit,SigmaError)

        Plots[1][1].bar(BarsPeriod, height=Ocurrence, width=Width, align="edge",
        facecolor=Color, edgecolor=Color, alpha=0.5)
        Plots[1][1].plot(PeriodRange, Gaussian_Dist(PeriodRange,AmpFit,SigmaFit,MeanFit), linestyle="--", color=Color, linewidth=1.5,
        label=labelFit)

    Plots[1][1].legend()
    SavePlot("PeriodDistribution", Name, Plots[0][1])

def BarsFreq_Month(Time_TIDs, Months_TIDs, Plots, Name):
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
    SavePlot("DayNightTIDs", Name, Plots[0][2])

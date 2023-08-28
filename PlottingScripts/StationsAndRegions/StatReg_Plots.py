from matplotlib.pyplot import figure, colorbar
from matplotlib import use

from lmfit.models import GaussianModel
from scipy.stats.mstats import mquantiles
import numpy as np

from DataScripts.CommonDictionaries import TerminatorsDict, DayNightColors
from PlottingScripts.SaveFunctions import SaveStationPlot, SaveRegionPlot

use("TkAgg")

# -------------------------------------------------------------------------------------------------------------------------------------------------

def CreateFiguresForStationsAndRegions(Name, Stat_or_Reg):
    # Create main figure for each analysis
    FigureOcurrenceHist = figure(1, figsize=(6, 6))
    FigurePeriodDists = figure(2, figsize=(6, 6))
    FigureMonthBars = figure(3, figsize=(6, 6))

    SubPlot_OcurrenceHist = FigureOcurrenceHist.add_subplot(111)
    SubPlot_PeriodDists = FigurePeriodDists.add_subplot(111)
    SubPlot_MonthBars = FigureMonthBars.add_subplot(111)

    # Setting titles
    if Stat_or_Reg == "Stat":
        Coord_String = f"\n{Name} Station"
    elif Stat_or_Reg == "Reg":
        Coord_String = f"\n{Name} Region"

    SubPlot_OcurrenceHist.set_title(Coord_String)
    SubPlot_PeriodDists.set_title(Coord_String)
    SubPlot_MonthBars.set_title(Coord_String)

    # Time Histogram labels
    SubPlot_OcurrenceHist.set_xlabel("Local Time (Hours)")
    SubPlot_OcurrenceHist.set_ylabel("Month")
    # Period Histogram labels
    SubPlot_PeriodDists.set_xlabel("TID Period (Minutes)")
    SubPlot_PeriodDists.set_ylabel("Probability Density")
    # Month frequencies labels
    SubPlot_MonthBars.set_ylabel("Number of events")

    return {"OCURR":(FigureOcurrenceHist, SubPlot_OcurrenceHist),
            "PERIOD":(FigurePeriodDists, SubPlot_PeriodDists),
            "DAY-NIGHT_BARS":(FigureMonthBars, SubPlot_MonthBars)}

# -------------------------------------------------------------------------------------------------------------------------------------------------


def Add_TimeMonthsHistogramToPlot(HistogramMonths, CMAP, NORM, Plots, RegionName, StationName, Stat_or_Reg):
    # Setting number of bins and time range for histogram
    TimeRange = (0.0, 24.0)
    MonthRange = (0, 12)
    # Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May",
                  "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = np.linspace(0.5, 11.5, 12, endpoint=True)
    Plots[1].set_yticks(MonthAxisData, MonthTicks)

    # Set the limits for Local Time and indexes for each Month
    extent = (*TimeRange, *MonthRange)
    timeTicks = np.arange(0, 25, 6)
    Plots[1].set_xlim(*TimeRange)
    Plots[1].set_ylim(*MonthRange)
    Plots[1].set_xticks(timeTicks)

    HistogramaImagen = Plots[1].imshow(HistogramMonths, cmap=CMAP, norm=NORM,
                                          interpolation="spline36", aspect="auto", origin="lower", extent=extent)
    colorbar(HistogramaImagen, ax=Plots[1], label="% Ocurrence")

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    NumMonthTerminator = np.linspace(0.0, 12.0, RiseHours.size)
    Plots[1].plot(RiseHours, NumMonthTerminator, "--k", linewidth=1.0)
    Plots[1].plot(SetHours, NumMonthTerminator, "--k", linewidth=1.0)

    Plots[0].tight_layout()
    if Stat_or_Reg == "Stat":
        SaveStationPlot("OcurrenceTIDs_", RegionName, StationName, Plots[0])
    elif Stat_or_Reg == "Reg":
        SaveRegionPlot("OcurrenceTIDs_", RegionName, Plots[0])


# -------------------------------------------------------------------------------------------------------------------------------------------------
# Same probability density function as used in lmfit module
# check GaussianModdel in https://lmfit.github.io/lmfit-py/builtin_models.html
def GaussianDist(x, A, mu, sigma): return (
    A/(sigma*(2.0*np.pi)**0.5))*np.exp(-0.5*((x-mu)/sigma)**2.0)


def Add_PeriodHistogramToPlot(Period, Time_TIDs, Months_TIDs, Plots, RegionName, StationName, Stat_or_Reg):
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
        PeriodHistogram, Edges, _ = Plots[1].hist(PeriodData, bins=PeriodBins, range=PeriodRange, density=True,
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
        Plots[1].plot(PeriodLinSampling, GaussianFitCurve, linestyle="--", color=Color, linewidth=1.5,
                         label=labelFit)

    Plots[1].legend()
    Plots[0].tight_layout()
    if Stat_or_Reg == "Stat":
        SaveStationPlot("PeriodDistribution_", RegionName,
                        StationName, Plots[0])
    elif Stat_or_Reg == "Reg":
        SaveRegionPlot("PeriodDistribution_", RegionName, Plots[0])

# -------------------------------------------------------------------------------------------------------------------------------------------------


def Add_BarsFreq_Month(Time_TIDs, Months_TIDs, Plots, RegionName, StationName, Stat_or_Reg):
    # Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May",
                  "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = np.linspace(0.5, 11.5, 12, endpoint=True)
    Plots[1].set_xticks(MonthAxisData, MonthTicks)

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(1, 2), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[0:SizeData:
                                    DivH_12], SetHours[0:SizeData:DivH_12]

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

    Plots[1].bar(x=MonthAxisData - 0.25, height=NumEvents_Day, width=0.25,
                    align="edge", edgecolor="k", facecolor="r", label="Day")
    Plots[1].bar(x=MonthAxisData, height=NumEvents_Night, width=0.25,
                    align="edge", edgecolor="k", facecolor="b", label="Night")

    Plots[1].legend()
    Plots[0].tight_layout()
    if Stat_or_Reg == "Stat":
        SaveStationPlot("DayNightTIDs_", RegionName, StationName, Plots[0])
    elif Stat_or_Reg == "Reg":
        SaveRegionPlot("DayNightTIDs_", RegionName, Plots[0])

from matplotlib.pyplot import subplots, colorbar
from matplotlib.ticker import LogFormatterSciNotation

from scipy.stats import iqr, skew, f, t, ttest_ind, shapiro, gamma, probplot, gaussian_kde
from scipy.optimize import root_scalar
from scipy.stats.mstats import mquantiles
import numpy as np
from lmfit.models import GaussianModel, PowerLawModel

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
    for i in range(Nplots):
        PowerVarSub[i][0].set_ylabel("TID Power (dTEC²)", fontsize=8)
        PowerVarSub[i][1].set_ylabel("IQR-Power (dTEC²)", fontsize=8)
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
    for i in range(Nplots):
        AmpVarSub[i][0].set_ylabel("AMA (dTEC)", fontsize=8)
        AmpVarSub[i][1].set_ylabel("AMA (dTEC)", fontsize=8)
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

    return {"POWER-VAR": (FigurePowerVar, PowerVarSub),
            "AMP-POWER": (FigureAmplitudePower, AmpPowSub),
            "AMP-VAR": (FigureAmplitudeVar, AmpVarSub),
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
                                     usecols=(0, 1), unpack=True, skiprows=1)
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
    Plots[1][Index].text(0.05, 0.95, f"A = {Best_A:.1f}\nExponent = {Best_k:.1f}\n"+r"$R^{{2}}$ = {0:.1f}".format(R2_Score),
                         horizontalalignment="left", verticalalignment="top", fontsize=9,
                         transform=Plots[1][Index].transAxes)

    Plots[1][Index].set_title(IndexName[Index])

    # And finally, add number of Day and Night events
    Plots[1][Index].text(0.95, 0.05, f"Day = {NumDay} Night = {NumNight}",
                         horizontalalignment="right", verticalalignment="bottom", fontsize=9,
                         transform=Plots[1][Index].transAxes)

# -------------------------------------------------------------------------------------------------------------------------------------------------

def Add_QuantityVarAnalysis(Quantity, Time_TIDs, Months_TIDs, Plots, Index, RegionName):
    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(0, 1), unpack=True, skiprows=1)
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
        MaskTime = np.where((Hour <= Time_TIDs) & (Time_TIDs <= Hour+2), True, False)
        AverageQuantity = Quantity[MaskTime]
        TwoHourQuantityQuantiles = mquantiles(AverageQuantity)

        LowerQ[n] = TwoHourQuantityQuantiles[0]
        Median[n] = TwoHourQuantityQuantiles[1]
        HigherQ[n] = TwoHourQuantityQuantiles[2]

    Plots[1][Index][0].fill_between(Hours+1, HigherQ, LowerQ, alpha=0.5, linewidth=0,
                                    color="black")
    Plots[1][Index][0].plot(Hours+1, LowerQ, color="black", linewidth=1, linestyle = "--")
    Plots[1][Index][0].plot(Hours+1, Median, color="black", linewidth=1, linestyle = "-")
    Plots[1][Index][0].plot(Hours+1, HigherQ, color="black", linewidth=1, linestyle = "--")

    # START ANALYSIS BY DATE DIVIDED IN DAY AND NIGHT ACTIVITY
    Months = np.array(list(range(1, 13, 1)))
    DayLowerQ = len(Months)*[0.0]
    DayMedian = len(Months)*[0.0]
    DayHigherQ = len(Months)*[0.0]          
    NightLowerQ = len(Months)*[0.0]
    NightMedian = len(Months)*[0.0]
    NightHigherQ = len(Months)*[0.0]

    for month in range(1, 13):
        Conds_month = (Months_TIDs == month)
        if Conds_month.any():
            month_index = month - 1
            Time_Conds_month = Time_TIDs[Conds_month]
            Quantity_Conds_month = Quantity[Conds_month]        

            # Filter for daytime events
            MaskDay = (RiseHours[month_index] <= Time_Conds_month) & (Time_Conds_month <= SetHours[month_index])
            Quantity_Day = Quantity_Conds_month[MaskDay]
            MonthDayQuantiles = mquantiles(Quantity_Day)

            DayLowerQ[month_index] = MonthDayQuantiles[0]
            DayMedian[month_index] = MonthDayQuantiles[1]
            DayHigherQ[month_index] = MonthDayQuantiles[2]

            # Filter for nighttime events
            MaskNight = (Time_Conds_month < RiseHours[month_index]) | (SetHours[month_index] < Time_Conds_month)
            Quantity_Night = Quantity_Conds_month[MaskNight]
            MonthNighQuantiles = mquantiles(Quantity_Night)

            NightLowerQ[month_index] = MonthNighQuantiles[0]
            NightMedian[month_index] = MonthNighQuantiles[1]
            NightHigherQ[month_index] = MonthNighQuantiles[2]

    Plots[1][Index][1].fill_between(Months, DayHigherQ, DayLowerQ, alpha=0.5, linewidth=0,
                                    color="red")
    Plots[1][Index][1].plot(Months, DayLowerQ, color="red", linewidth=1, linestyle = "--")
    Plots[1][Index][1].plot(Months, DayMedian, color="red", linewidth=1, linestyle = "-")
    Plots[1][Index][1].plot(Months, DayHigherQ, color="red", linewidth=1, linestyle = "--")

    Plots[1][Index][2].fill_between(Months, NightHigherQ, NightLowerQ, alpha=0.5, linewidth=0,
                                    color="blue")
    Plots[1][Index][2].plot(Months, NightLowerQ, color="blue", linewidth=1, linestyle="--")
    Plots[1][Index][2].plot(Months, NightMedian, color="blue", linewidth=1, linestyle="-")
    Plots[1][Index][2].plot(Months, NightHigherQ, color="blue", linewidth=1, linestyle="--")

    # Set title to middle column
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
                                     usecols=(0, 1), unpack=True, skiprows=1)
    NumMonthTerminator = np.linspace(0.0, 12.0, RiseHours.size)
    Plots[1][Index].plot(RiseHours, NumMonthTerminator, "--k", linewidth=1.0)
    Plots[1][Index].plot(SetHours, NumMonthTerminator, "--k", linewidth=1.0)

    Plots[1][Index].set_title(IndexName[Index])

# -------------------------------------------------------------------------------------------------------------------------------------------------

def ComparisonOfDayNightVarianceMean(DayTIDsPeriods, NightTIDsPeriods, RegionName):
    large_std = max(DayTIDsPeriods.std(ddof = 1), NightTIDsPeriods.std(ddof = 1))
    small_std = min(DayTIDsPeriods.std(ddof = 1), NightTIDsPeriods.std(ddof = 1))

    if large_std == DayTIDsPeriods.std():
        N1 = DayTIDsPeriods.size
        N2 = NightTIDsPeriods.size
    else:
        N1 = NightTIDsPeriods.size
        N2 = DayTIDsPeriods.size

    F = (large_std/small_std) ** 2.0
    alpha = 0.05
    F_crit_lower_value = f.ppf(0.5*alpha, N1 - 1, N2 - 1)
    F_crit_upper_value = f.ppf(1 - 0.5*alpha, N1 - 1, N2 - 1)

    print("\nF-test for " + RegionName)
    print(f"{alpha=}")
    print(f"F statistic= {F:.5f}")
    print(f"F lower critic value= {F_crit_lower_value:.5f}")
    print(f"F upper critic value= {F_crit_upper_value:.5f}")

    if F < F_crit_lower_value or F > F_crit_upper_value:
        print("Reject the F-test null hypothesis. Day and Night variances are significantly different")
        ttest_results = ttest_ind(DayTIDsPeriods, NightTIDsPeriods, equal_var=False)
    else:
        print("Don't reject the F-test null hypothesis. Day and Night variances are basically the same")
        ttest_results = ttest_ind(DayTIDsPeriods, NightTIDsPeriods, equal_var=True)
    T_crit_value = t.ppf(1.0 - 0.5*alpha, ttest_results.df)

    print("T-test for " + RegionName)
    print(f"T statistic= {ttest_results.statistic:.5f}")
    print(f"T critic value= {T_crit_value:.5f}")                                
    if abs(ttest_results.statistic) > T_crit_value:
        print("Reject the T-test null hypothesis. Day and Night means are significantly different")
    else:
        print("Don't reject the T-test null hypothesis. Day and Night means are basically the same")
    print("\n")

# Compute S-W for a given Gamma shape parameter and sample size
def ComputeShapiroWilk(gshape=20, n=50):
    data = gamma.ppf((np.arange(1, n + 1) / (n + 1)), a=gshape, scale=1)
    stat, p_value = shapiro(data)
    return stat, p_value

# Find shape parameter that corresponds to a particular p-value
def FindShape(n, alpha):
    def objective(gshape):
        _, p_value = ComputeShapiroWilk(gshape, n)
        return p_value - alpha
    
    result = root_scalar(objective, bracket=[0.01, 100], method='brentq')
    return result.root

# Find W statistic for given n and alpha
def FindCriticalW(n, alpha):
    s = FindShape(n, alpha)
    stat, _ = ComputeShapiroWilk(s, n)
    return stat

def CheckNormality(DayTIDsPeriods, NightTIDsPeriods, RegionName):
    alpha = 0.05
    day_normality = shapiro(DayTIDsPeriods)
    CriticalWDay = FindCriticalW(DayTIDsPeriods.size, alpha)
    night_normality = shapiro(NightTIDsPeriods)
    CriticalWNight = FindCriticalW(NightTIDsPeriods.size, alpha)

    print("\nShapiro-Wilk test for " + RegionName)
    print(f"{alpha=}")
    print(f"Day statistic= {day_normality[0]:.5f}")
    print(f"Day critical statistic= {CriticalWDay:.5f}")
    print(f"Day p-value= {day_normality[1]:.5f}")
    print(f"Night statistic= {night_normality[0]:.5f}")
    print(f"Night critical statistic= {CriticalWNight:.5f}")
    print(f"Night p-value= {night_normality[1]:.5f}")
    if day_normality[0] > CriticalWDay:
        print("Don't reject Shapiro null hypothesis. Day TIDs period distribute as normal")
    else:
        print("Reject Shapiro null hypothesis. Day TIDs period do not distribute as normal")

    if night_normality[0] > CriticalWNight:
        print("Don't reject Shapiro null hypothesis. Night TIDs period distribute as normal")
    else:
        print("Reject Shapiro null hypothesis. Night TIDs period do not distribute as normal")


    FigNormality, SubplotsNorm = subplots(num = 8, nrows = 1, ncols = 2, sharey = "all")

    rDay = probplot(DayTIDsPeriods, plot = SubplotsNorm[0])[1][2]
    rNight = probplot(NightTIDsPeriods, plot = SubplotsNorm[1])[1][2]

    bbox_props = dict(boxstyle='round', facecolor='white', alpha=1.0)

    FigNormality.suptitle(RegionName)
    SubplotsNorm[0].set_title("Day")
    SubplotsNorm[0].grid()
    SubplotsNorm[1].set_title("Night")
    SubplotsNorm[1].grid()
    SubplotsNorm[1].set_ylabel("")

    SubplotsNorm[0].text(0.2, 0.8, r'$R^{{2}}$={0:.3f}'.format(rDay**2.0), horizontalalignment='center',
    verticalalignment='center', transform=SubplotsNorm[0].transAxes, bbox = bbox_props)
    SubplotsNorm[1].text(0.2, 0.8, r'$R^{{2}}$={0:.3f}'.format(rNight**2.0), horizontalalignment='center',
    verticalalignment='center', transform=SubplotsNorm[1].transAxes, bbox = bbox_props)

    FigNormality.tight_layout()
    FigNormality.savefig(f"../Results/{RegionName}/QQplots_{RegionName}.png")
    FigNormality.clear()

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Same probability density function as used in lmfit module
# check GaussianModel in https://lmfit.github.io/lmfit-py/builtin_models.html
def GaussianDist(x, A, mu, sigma): 
    return (A/(sigma*(2.0*np.pi)**0.5))*np.exp(-0.5*((x-mu)/sigma)**2.0)

def RemovePeriodOutliers(SamplePeriods):
    PeriodIQR = iqr(SamplePeriods)
    PeriodQ1, PeriodQ3 = np.quantile(SamplePeriods, 0.25), np.quantile(SamplePeriods, 0.75)
    PeriodValues_Outliers = np.argwhere((SamplePeriods < PeriodQ1 - 1.5*PeriodIQR) | (SamplePeriods > PeriodQ3 + 1.5*PeriodIQR))[:,0]
    OutliersSamplePeriods = np.copy(SamplePeriods[PeriodValues_Outliers])
    PeriodValues_NotOutliers = np.argwhere((SamplePeriods >= PeriodQ1 - 1.5*PeriodIQR) & (SamplePeriods <= PeriodQ3 + 1.5*PeriodIQR))[:,0]
    SamplePeriods = SamplePeriods[PeriodValues_NotOutliers]
    SamplePeriods.sort()

    return SamplePeriods, OutliersSamplePeriods

def Add_PeriodHistogramToPlot(Period, Time_TIDs, Months_TIDs, Plots, Index, RegionName):
    Period = 60.0*Period

    # Extracting rise and set hours for each region
    RiseHours, SetHours = np.loadtxt(TerminatorsDict[RegionName], dtype=np.float64,
                                     usecols=(0, 1), unpack=True, skiprows=1)
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

    # Filter events that could be considered outliers given the saved periods
    DayTIDsPeriods, OutliersDay = RemovePeriodOutliers(DayTIDsPeriods)
    NightTIDsPeriods, OutliersNight = RemovePeriodOutliers(NightTIDsPeriods)

    CheckNormality(DayTIDsPeriods, NightTIDsPeriods, RegionName)
    ComparisonOfDayNightVarianceMean(DayTIDsPeriods, NightTIDsPeriods, RegionName)

    for n, Periods_Name in enumerate(zip([(DayTIDsPeriods, OutliersDay), (NightTIDsPeriods, OutliersNight)], ["Day", "Night"])):
        # Extract the sample of periods
        SamplePeriods = np.copy(Periods_Name[0][0])
        # Extract color
        Color = DayNightColors[Periods_Name[1]]

        PeriodRange = (SamplePeriods.min(), SamplePeriods.max())    
        # Setting number of bins by using the Scotts's rule
        BinWidth = 3.5 * SamplePeriods.std(ddof = 1) / (SamplePeriods.size ** (1/3))
        PeriodBins = int( (PeriodRange[1] - PeriodRange[0])/BinWidth )

        # Adding density histogram of period data
        Plots[1][Index][n].hist(SamplePeriods, bins=PeriodBins, range=PeriodRange, density=True,
                                facecolor=Color, edgecolor="None", alpha=0.5)[0]

        # Declare a Gaussian KDE to find a numerical distribution
        GaussianKDE = gaussian_kde(SamplePeriods, bw_method = "scott")
        GaussianKDE_eval = GaussianKDE.evaluate(SamplePeriods)

        # Declaring an Gaussian Model as the proposed theoretical distribution
        GaussianToFit = GaussianModel()
        # Getting mean, deviation and skewness of period data and max value of Ocurrence
        Mean, Deviation = SamplePeriods.mean(), SamplePeriods.std(ddof = 1)
        MaxValue = GaussianKDE_eval.max()
        # Setting parameters
        ParametersExpGaussian = GaussianToFit.make_params(amplitude=MaxValue, center=Mean, sigma=Deviation)
        # Calculate best fit
        GaussianFitResult = GaussianToFit.fit(GaussianKDE_eval, ParametersExpGaussian, x = SamplePeriods)

        # Extracting optimal parameters for gaussian fit
        ParamsResults = GaussianFitResult.params
        AmpFit = ParamsResults["amplitude"].value
        MeanFit, MeanError = ParamsResults["center"].value, ParamsResults["center"].stderr
        SigmaFit, SigmaError = ParamsResults["sigma"].value, ParamsResults["sigma"].stderr

        # Create string sequence to show optimal mean and deviation values for the input data
        labelFit = Periods_Name[1]+" fit"+"\n"+r"$\mu$={0:.1f}$\pm${1:.2f}".format(MeanFit, MeanError)
        labelFit += "\n"
        labelFit += r"$\sigma$={0:.1f}$\pm${1:.2f}".format(SigmaFit, SigmaError)

        # Create theoretical distribution given the optimal values
        PeriodLinSampling = np.linspace(PeriodRange[0], 60.0, 200, endpoint=True)
        GaussianFitCurve = GaussianDist(PeriodLinSampling, AmpFit, MeanFit, SigmaFit)

        # Adding gaussian kde curve
        #Plots[1][Index][n].plot(SamplePeriods, GaussianKDE.evaluate(SamplePeriods), 
        #                        linestyle="--", color="black", 
        #                        linewidth=1.5, alpha = 0.65,
        #                        label=f"KDE\nh={GaussianKDE.factor:.3f}")
        # Adding gaussian curve by using the optimal parameters
        Plots[1][Index][n].plot(PeriodLinSampling, GaussianFitCurve, linestyle="--", color=Color, linewidth=1.5,
                             label=labelFit)

        # Plot outliers as dots in x-axis
        Plots[1][Index][n].scatter(Periods_Name[0][1], Periods_Name[0][1].size*[0.0025], 
                                   c = "black", s = 4.0, marker = "o")  
        
        Plots[1][Index][n].set_xlim(15.0, 60.0)
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
                                     usecols=(0, 1), unpack=True, skiprows=1)
    SizeData = RiseHours.size
    DivH_12 = SizeData//12
    RiseHours, SetHours = RiseHours[::DivH_12], SetHours[::DivH_12]

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

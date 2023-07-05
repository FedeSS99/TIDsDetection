from PlottingScripts.SaveFunctions import SaveAllRegionPlot
from numpy import linspace, arange

import warnings

def FormatAndSave_AllRegionPlots(Nplots, PLOTS, ListBoxPlots, ListLegendsBoxPlots):
    # ------ APPLY FORMAT TO OCURRENCE FIGURE ------
    # Setting number of bins and time range for histogram
    TimeRange = (0.0, 24.0)
    MonthRange = (0, 12)
    # Setting y ticks with months names
    MonthStrTicks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthTicksOcurr = linspace(0.5, 11.5, 12, endpoint=True)
    # Set the limits for Local Time and indexes for each Month
    timeTicks = arange(0, 25, 3)

    # Desactivate UserWarning, expected output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PLOTS["OCURR"][0].tight_layout()
    for p in range(Nplots):
        PLOTS["OCURR"][1][p].set_yticks(MonthTicksOcurr, MonthStrTicks)

        PLOTS["OCURR"][1][p].set_xlim(*TimeRange)
        PLOTS["OCURR"][1][p].set_ylim(*MonthRange)
        PLOTS["OCURR"][1][p].set_xticks(timeTicks)

        SubplotBox = PLOTS["OCURR"][1][p].get_position()
        PLOTS["OCURR"][1][p].set_position([SubplotBox.x0, SubplotBox.y0,
                               0.75*SubplotBox.width, SubplotBox.height])
    SaveAllRegionPlot("OcurrenceTIDs", PLOTS["OCURR"][0])

    # ------ APPLY FORMAT TO PERIOD DISTRIBUTION FIGURE ------
    PLOTS["PERIOD"][0].tight_layout()
    MinDistPer_Plots, MaxDistPer_Plots = [], []
    for p in range(Nplots):      
        Y_MIN, Y_MAX = PLOTS["PERIOD"][1][p].get_ylim()
        MinDistPer_Plots.append(Y_MIN)
        MaxDistPer_Plots.append(Y_MAX)

    Y_MIN, Y_MAX = min(MinDistPer_Plots), max(MaxDistPer_Plots)
    for p in range(Nplots):
        PLOTS["PERIOD"][1][p].set_ylim(Y_MIN, Y_MAX)

    SaveAllRegionPlot("PeriodDistribution", PLOTS["PERIOD"][0])

    # ------ APPLY FORMAT TO DAY-NIGHT BARS FIGURE ------
    PLOTS["DAY-NIGHT_BARS"][0].tight_layout()
    MinBar_Plots, MaxBar_Plots = [], []
    for p in range(Nplots):
        SubplotBox = PLOTS["DAY-NIGHT_BARS"][1][p].get_position()
        PLOTS["DAY-NIGHT_BARS"][1][p].set_position([SubplotBox.x0, SubplotBox.y0,
                                                    0.8*SubplotBox.width, SubplotBox.height])

        Y_MIN, Y_MAX = PLOTS["DAY-NIGHT_BARS"][1][p].get_ylim()
        MinBar_Plots.append(Y_MIN)
        MaxBar_Plots.append(Y_MAX)

    Y_MIN, Y_MAX = min(MinBar_Plots), max(MaxBar_Plots)
    for p in range(Nplots):
        PLOTS["DAY-NIGHT_BARS"][1][p].set_ylim(Y_MIN, Y_MAX)

    SaveAllRegionPlot("DayNightTIDs", PLOTS["DAY-NIGHT_BARS"][0])

    # ------ APPLY FORMAT TO POWER VARIABILITY FIGURE ------
    # Apply logaritmic scale to Power variance plot and save the figure
    PLOTS["POWER_VAR"][1].set_yscale("log")
    SubplotBox = PLOTS["POWER_VAR"][1].get_position()
    PLOTS["POWER_VAR"][1].set_position([SubplotBox.x0, SubplotBox.y0,
                                        0.925*SubplotBox.width, SubplotBox.height])
    PLOTS["POWER_VAR"][0].legend(ListBoxPlots, ListLegendsBoxPlots,
                                 loc="upper right", fancybox=True, shadow=True, bbox_to_anchor=(1.0, 0.5))
    SaveAllRegionPlot("PowerDistributionStations", PLOTS["POWER_VAR"][0])


    # ------ APPLY FORMAT TO AMPLITUDE VARIABILITY FIGURE ------
    # Create Hours and Months ticks to use change the labels in Amplitude variance
    # plots and save the figure
    HourTicks = list(range(0, 25, 4))
    HourStrTicks = [str(num) for num in HourTicks]
    MonthTicks = list(range(1,13))
    MinAmps_Plots, MaxAmps_Plots = [], []
    for p in range(Nplots-1):
        Y_MIN, Y_MAX = PLOTS["AMP_VAR"][1][p][0].get_ylim()
        MinAmps_Plots.append(Y_MIN)
        MaxAmps_Plots.append(Y_MAX)

    # Set the same min and max value for all Amplitude variance plots
    Y_MIN, Y_MAX = min(MinAmps_Plots), max(MaxAmps_Plots)
    for p in range(Nplots):
        for l in range(3):
            PLOTS["AMP_VAR"][1][p][l].set_ylim(Y_MIN, Y_MAX)

            if l > 0:
                PLOTS["AMP_VAR"][1][p][l].set_yticks(ticks=[], labels=[])
                PLOTS["AMP_VAR"][1][p][l].set_xticks(ticks=MonthTicks, labels=MonthStrTicks)

    # Fix date formatting in x axis
    PLOTS["AMP_VAR"][0].autofmt_xdate(bottom=0.1, rotation=45)

    # Setting x ticks within 24 hours, with zero rotation and centered
    PLOTS["AMP_VAR"][1][Nplots - 1][0].set_xticks(HourTicks, HourStrTicks, rotation=0, ha="center")
    PLOTS["AMP_VAR"][1][Nplots - 1][0].set_xlim(0.0, 24.0)

    # Modify the fixed positions and dimensiones of the subplots for
    # better visualizationNplots
    PLOTS["AMP_VAR"][0].tight_layout()

    SaveAllRegionPlot("AmplitudeVariations", PLOTS["AMP_VAR"][0]) 


    # ------ APPLY FORMAT TO AMPLITUDE VS POWER FIGURE ------
    PLOTS["AMP_POWER"][0].tight_layout()

    # Apply logaritmic scale to x-axis and y-axis in each Amplitude-Power plot
    # and also set the same y-axis limits in all Amplitude-Power plots
    MinAmpPow_Plots, MaxAmpPow_Plots = [], []
    for p in range(Nplots):
        SubplotBox = PLOTS["AMP_POWER"][1][p].get_position()
        PLOTS["AMP_POWER"][1][p].set_position([SubplotBox.x0, SubplotBox.y0,
                                               0.7*SubplotBox.width, SubplotBox.height])

        Y_MIN, Y_MAX = PLOTS["AMP_POWER"][1][p].get_ylim()
        MinAmpPow_Plots.append(Y_MIN)
        MaxAmpPow_Plots.append(Y_MAX)
        PLOTS["AMP_POWER"][1][p].set_yscale("log", subs=None)

    PLOTS["AMP_POWER"][1][Nplots-1].set_xscale("log", subs=None)

    Y_MIN, Y_MAX = min(MinAmpPow_Plots), max(MaxAmpPow_Plots)
    for p in range(Nplots):
        # Desactivate UserWarning, expected output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PLOTS["AMP_POWER"][1][p].set_ylim(Y_MIN, Y_MAX)

    # Fixing position of legends box outside the subplots
    PLOTS["AMP_POWER"][0].legend(loc="center right", bbox_to_anchor=(1, 0.5),
                                 fancybox=True, shadow=True)

    SaveAllRegionPlot("AmplitudePowerRegions", PLOTS["AMP_POWER"][0])

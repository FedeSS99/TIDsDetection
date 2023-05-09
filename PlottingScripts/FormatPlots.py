from PlottingScripts.SaveFunctions import SaveAllRegionPlot
from numpy import linspace, arange

def FormatAndSave_AllRegionPlots(Nplots, PLOTS, ListBoxPlots, ListLegendsBoxPlots):
    # ------ APPLY FORMAT TO OCURRENCE FIGURE ------
    # Setting number of bins and time range for histogram
    TimeRange = (0.0, 24.0)
    MonthRange = (0, 12)
    # Setting y ticks with months names
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May",
                  "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MonthAxisData = linspace(0.5, 11.5, 12, endpoint=True)

    # Set the limits for Local Time and indexes for each Month
    
    timeTicks = arange(0, 25, 3)

    PLOTS["OCURR"][0].tight_layout()
    for p in range(Nplots):
        PLOTS["OCURR"][1][p].set_yticks(MonthAxisData, MonthTicks)
        PLOTS["OCURR"][1][p].set_yticklabels(MonthTicks, rotation=45)

        PLOTS["OCURR"][1][p].set_xlim(*TimeRange)
        PLOTS["OCURR"][1][p].set_ylim(*MonthRange)
        PLOTS["OCURR"][1][p].set_xticks(timeTicks)

        SubplotBox = PLOTS["OCURR"][1][p].get_position()
        PLOTS["OCURR"][1][p].set_position([SubplotBox.x0, SubplotBox.y0,
                               0.85*SubplotBox.width, SubplotBox.height])
    SaveAllRegionPlot("OcurrenceTIDs", PLOTS["OCURR"][0])

    # ------ APPLY FORMAT TO PERIOD DISTRIBUTION FIGURE ------
    PLOTS["PERIOD"][0].tight_layout()
    SaveAllRegionPlot("PeriodDistribution", PLOTS["PERIOD"][0])

    # ------ APPLY FORMAT TO POWER VARIABILITY FIGURE ------
    PLOTS["DAY-NIGHT_BARS"][0].tight_layout()
    SaveAllRegionPlot("DayNightTIDs", PLOTS["DAY-NIGHT_BARS"][0])

    # ------ APPLY FORMAT TO POWER VARIABILITY FIGURE ------
    # Apply logaritmic scale to Power variance plot and save the figure
    PLOTS["POWER_VAR"][1].set_yscale("log")
    SubplotBox = PLOTS["POWER_VAR"][1].get_position()
    PLOTS["POWER_VAR"][1].set_position([SubplotBox.x0, SubplotBox.y0,
                               SubplotBox.width*0.9, SubplotBox.height])
    PLOTS["POWER_VAR"][0].legend(ListBoxPlots, ListLegendsBoxPlots,
                        loc="upper right", fancybox=True, shadow=True, bbox_to_anchor=(1.0, 0.5))
    SaveAllRegionPlot("PowerDistributionStations", PLOTS["POWER_VAR"][0])


    # ------ APPLY FORMAT TO AMPLITUDE VARIABILITY FIGURE ------
    # Create Hours and Months ticks to use change the labels in Amplitude variance
    # plots and save the figure
    MonthTicks = ["Jan", "Feb", "Mar", "Apr", "May",
                  "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    HOUR_TICKS = [i for i in range(0, 25, 4)]
    HOUR_STR_TICKS = [str(num) for num in HOUR_TICKS]
    MonthAxisData = linspace(1.0, 12.0, 12, endpoint=True)
    MinAmps_Plots, MaxAmps_Plots = [], []
    for p in range(Nplots-1):
        PLOTS["AMP_VAR"][1][p][0].set_xticks(ticks=[], labels=[])
        PLOTS["AMP_VAR"][1][p][1].set_xticks(ticks=[], labels=[])

    # Setting x ticks within 24 hours
    PLOTS["AMP_VAR"][1][Nplots - 1][0].set_xticks(ticks=HOUR_TICKS, labels=HOUR_STR_TICKS)
    PLOTS["AMP_VAR"][1][Nplots - 1][0].set_xlim(0.0, 24.0)

    # Setting x ticks with months names
    PLOTS["AMP_VAR"][1][Nplots - 1][1].set_xticks(ticks=MonthAxisData, labels=MonthTicks)
    PLOTS["AMP_VAR"][1][Nplots - 1][1].set_xlim(1.0, 13.0)

    # Align to the right and rotate the x-axis labels of the bottom 2nd column
    for label in PLOTS["AMP_VAR"][1][Nplots - 1][1].get_xticklabels():
        label.set_horizontalalignment('left')
    PLOTS["AMP_VAR"][1][Nplots - 1][1].set_xticklabels(MonthTicks, rotation=-45)

    Y_MIN, Y_MAX = PLOTS["AMP_VAR"][1][p][0].get_ylim()
    MinAmps_Plots.append(Y_MIN)
    MaxAmps_Plots.append(Y_MAX)

    # Set the same min and max value for all Amplitude variance plots
    Y_MIN, Y_MAX = min(MinAmps_Plots), max(MaxAmps_Plots)
    for p in range(Nplots):
        for l in range(2):
            PLOTS["AMP_VAR"][1][p][l].set_ylim(Y_MIN, Y_MAX)
        PLOTS["AMP_VAR"][1][p][1].set_yticks(ticks=[], labels=[])

    PLOTS["AMP_VAR"][0].tight_layout()
    SaveAllRegionPlot("AmplitudeVariations", PLOTS["AMP_VAR"][0]) 


    # ------ APPLY FORMAT TO AMPLITUDE VS POWER FIGURE ------
    # Apply logaritmic scale to x-axis and y-axis in Amplitude-Power plot
    for p in range(Nplots):
        PLOTS["AMP_POWER"][1][p].set_yscale("log", subs=None)
    PLOTS["AMP_POWER"][1][Nplots-1].set_xscale("log", subs=None)

    PLOTS["AMP_POWER"][0].tight_layout()

    # Fixing position of legends box outside the subplots
    SaveAllRegionPlot("AmplitudePowerRegions", PLOTS["AMP_POWER"][0])

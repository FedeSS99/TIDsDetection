from PlottingScripts.SaveFunctions import SaveAllRegionPlot
from numpy import linspace

def FormatAndSave_AllRegionPlots(Nplots, PLOTS, ListBoxPlots, ListLegendsBoxPlots):
    # ------ APPLY FORMAT TO OCURRENCE FIGURE ------
    SaveAllRegionPlot("OcurrenceTIDs", PLOTS["OCURR"][0])

    # ------ APPLY FORMAT TO PERIOD DISTRIBUTION FIGURE ------
    SaveAllRegionPlot("PeriodDistribution", PLOTS["PERIOD"][0])

    # ------ APPLY FORMAT TO POWER VARIABILITY FIGURE ------
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
    MonthAxisData = linspace(1.0, 12.0, 12, endpoint=True)
    MinAmps_Plots, MaxAmps_Plots = [], []
    for p in range(Nplots):
        if p < Nplots - 1:
            PLOTS["AMP_VAR"][1][p][0].set_xticks(ticks=[], labels=[])
            PLOTS["AMP_VAR"][1][p][1].set_xticks(ticks=[], labels=[])

        if p == Nplots - 1:
            # Setting x ticks within 24 hours
            PLOTS["AMP_VAR"][1][p][0].set_xticks(
                ticks=HOUR_TICKS, labels=HOUR_TICKS)
            # Setting x ticks with months names
            PLOTS["AMP_VAR"][1][p][1].set_xticks(
                ticks=MonthAxisData, labels=MonthTicks)

            # Align to the right and rotate the x-axis labels of the bottom 2nd column
            for label in PLOTS["AMP_VAR"][1][p][1].get_xticklabels():
                label.set_horizontalalignment('left')
            PLOTS["AMP_VAR"][1][p][1].set_xticklabels(
                MonthTicks, rotation=-45)

        Y_MIN, Y_MAX = PLOTS["AMP_VAR"][1][p][0].get_ylim()
        MinAmps_Plots.append(Y_MIN)
        MaxAmps_Plots.append(Y_MAX)

    # Set the same min and max value for all Amplitude variance plots
    Y_MIN, Y_MAX = min(MinAmps_Plots), max(MaxAmps_Plots)
    for p in range(Nplots):
        for l in range(2):
            PLOTS["AMP_VAR"][1][p][l].set_ylim(Y_MIN, Y_MAX)
        PLOTS["AMP_VAR"][1][p][1].set_yticks(ticks=[], labels=[])
    SaveAllRegionPlot("AmplitudeVariations", PLOTS["AMP_VAR"][0])


    # ------ APPLY FORMAT TO AMPLITUDE VS POWER FIGURE ------
    # Apply logaritmic scale to x-axis and y-axis in Amplitude-Power plot
    for p in range(Nplots):
        PLOTS["AMP_POWER"][1][p].set_yscale("log", subs=None)
    PLOTS["AMP_POWER"][1][Nplots-1].set_xscale("log", subs=None)

    # Fixing position of legends box outside the subplots
    SaveAllRegionPlot("AmplitudePowerRegions", PLOTS["AMP_POWER"][0])

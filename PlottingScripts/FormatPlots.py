from PlottingScripts.SaveFunctions import SaveAllRegionPlot
from DataScripts.CommonDictionaries import IndexName
from numpy import linspace, arange

import warnings

def FormatAndSave_AllRegionPlots(Nplots, PLOTS):
    print("Setting final format for all-region figures...", end="")

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

    for QUANT_VAR, NAME_VAR in zip(["OCURR", "POWER-MED"], ["OcurrenceTIDs", "PowerMedian"]):
        # Desactivate UserWarning, expected output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PLOTS[QUANT_VAR][0].tight_layout()
        for p in range(Nplots):
            PLOTS[QUANT_VAR][1][p].set_yticks(MonthTicksOcurr, MonthStrTicks)

            PLOTS[QUANT_VAR][1][p].set_xlim(*TimeRange)
            PLOTS[QUANT_VAR][1][p].set_ylim(*MonthRange)
            PLOTS[QUANT_VAR][1][p].set_xticks(timeTicks)

            SubplotBox = PLOTS[QUANT_VAR][1][p].get_position()
            PLOTS[QUANT_VAR][1][p].set_position([SubplotBox.x0, SubplotBox.y0, 0.75*SubplotBox.width, SubplotBox.height])
        SaveAllRegionPlot(NAME_VAR, PLOTS[QUANT_VAR][0])

    # ------ APPLY FORMAT TO PERIOD DISTRIBUTION FIGURE ------
    PLOTS["PERIOD"][0].tight_layout()
    for Index in range(PLOTS["PERIOD"][1].shape[0]):
        SubplotBox = PLOTS["PERIOD"][1][Index][0].get_position()
        PLOTS["PERIOD"][0].text(0.525, SubplotBox.y0 + SubplotBox.height + 0.0125, IndexName[Index], ha="center", va="center")

    SaveAllRegionPlot("PeriodDistribution", PLOTS["PERIOD"][0])

    # ------ APPLY FORMAT TO DAY-NIGHT BARS FIGURE ------
    PLOTS["DAY-NIGHT-BARS"][0].tight_layout()
    MinBar_Plots, MaxBar_Plots = [], []
    for p in range(Nplots):
        SubplotBox = PLOTS["DAY-NIGHT-BARS"][1][p].get_position()
        PLOTS["DAY-NIGHT-BARS"][1][p].set_position([SubplotBox.x0, SubplotBox.y0,
                                                    0.8*SubplotBox.width, SubplotBox.height])

        Y_MIN, Y_MAX = PLOTS["DAY-NIGHT-BARS"][1][p].get_ylim()
        MinBar_Plots.append(Y_MIN)
        MaxBar_Plots.append(Y_MAX)

    Y_MIN, Y_MAX = min(MinBar_Plots), max(MaxBar_Plots)
    for p in range(Nplots):
        PLOTS["DAY-NIGHT-BARS"][1][p].set_ylim(Y_MIN, Y_MAX)

    SaveAllRegionPlot("DayNightTIDs", PLOTS["DAY-NIGHT-BARS"][0])

    # ------ APPLY FORMAT TO AMPLITUDE AND POWER VARIABILITY FIGURES ------
    for QUANT_VAR, NAME_VAR in zip(["POWER-VAR", "AMP-VAR"], ["Power", "Amplitude"]):
        # Create Hours and Months ticks to use change the labels in Amplitude variance
        # plots and save the figure
        HourTicks = list(range(0, 25, 4))
        HourStrTicks = [str(num) for num in HourTicks]
        MonthTicks = list(range(1,13))

        # Run over every row, set the x limits for local time analysis and
        # months ticks for quantity dispersion and wind speed profiles
        for p in range(Nplots):
            PLOTS[QUANT_VAR][1][p][1].set_xlim(1,13)
            PLOTS[QUANT_VAR][1][p][2].set_xlim(1,13)
            
            PLOTS[QUANT_VAR][1][p][1].set_xticks(ticks=MonthTicks, labels=MonthStrTicks, rotation = 90, ha="center")
            PLOTS[QUANT_VAR][1][p][2].set_xticks(ticks=MonthTicks, labels=MonthStrTicks, rotation = 90, ha="center")

            PLOTS[QUANT_VAR][2][p][0].tick_params(axis="y", which="both", right=False, labelright=False)
            PLOTS[QUANT_VAR][1][p][2].tick_params(axis="y", which="both", left=False, labelleft=False)

        # Setting x ticks within 24 hours, with zero rotation and centered
        PLOTS[QUANT_VAR][1][Nplots - 1][0].set_xticks(HourTicks, HourStrTicks, rotation=0, ha="center")
        PLOTS[QUANT_VAR][1][Nplots - 1][0].set_xlim(0.0, 24.0)

        # Modify the fixed positions and dimensiones of the subplots for
        # better visualizationNplots
        PLOTS[QUANT_VAR][0].tight_layout()

        # Reduce the separation between day and night analysis subplots to zero
        for p in range(Nplots):
            SubplotBoxDay = PLOTS[QUANT_VAR][1][p][1].get_position()
            SubplotBoxNight = PLOTS[QUANT_VAR][1][p][2].get_position()

            SeparationDayNightPlots = SubplotBoxNight.x0 - (SubplotBoxDay.x0 + SubplotBoxDay.width)
            HalfSeparation = 0.4*SeparationDayNightPlots

            PLOTS[QUANT_VAR][1][p][1].set_position([SubplotBoxDay.x0, SubplotBoxDay.y0,
                                                    SubplotBoxDay.width + HalfSeparation, SubplotBoxDay.height])
            PLOTS[QUANT_VAR][1][p][2].set_position([SubplotBoxNight.x0-HalfSeparation, SubplotBoxNight.y0,
                                                    SubplotBoxNight.width+HalfSeparation, SubplotBoxNight.height])

        # Set equal y max limit for local time and month analysis
        DictYLims = dict(
            LocalTime = 0.0,
            Power = 0.0,
            WindSpeed = 0.0
        )
        for p in range(Nplots):
            MaxValPowerLT = PLOTS[QUANT_VAR][1][p][0].get_ylim()[1]
            MaxValPower = max([PLOTS[QUANT_VAR][1][p][n].get_ylim()[1] for n in range(1,3)])
            MaxValWindSpeed = max([PLOTS[QUANT_VAR][2][p][n].get_ylim()[1] for n in range(2)])

            DictYLims["LocalTime"] = max(DictYLims["LocalTime"], MaxValPowerLT)
            DictYLims["Power"] = max(DictYLims["Power"], MaxValPower)
            DictYLims["WindSpeed"] = max(DictYLims["WindSpeed"], MaxValWindSpeed)

        for p in range(Nplots):
            PLOTS[QUANT_VAR][1][p][0].set_ylim(top=DictYLims["LocalTime"])
            for m, n in enumerate(range(1,3)):
                PLOTS[QUANT_VAR][1][p][n].set_ylim(top=DictYLims["Power"])
                PLOTS[QUANT_VAR][2][p][m].set_ylim(top=DictYLims["WindSpeed"])

        SaveAllRegionPlot(f"{NAME_VAR}Variations", PLOTS[QUANT_VAR][0]) 


    # ------ APPLY FORMAT TO AMPLITUDE VS POWER FIGURE ------
    PLOTS["AMP-POWER"][0].tight_layout()

    # Apply logaritmic scale to x-axis and y-axis in each Amplitude-Power plot
    # and also set the same y-axis limits in all Amplitude-Power plots
    MinAmpPow_Plots, MaxAmpPow_Plots = [], []
    for p in range(Nplots):
        SubplotBox = PLOTS["AMP-POWER"][1][p].get_position()
        PLOTS["AMP-POWER"][1][p].set_position([1.15*SubplotBox.x0, SubplotBox.y0,
                                               0.7*SubplotBox.width, SubplotBox.height])

        Y_MIN, Y_MAX = PLOTS["AMP-POWER"][1][p].get_ylim()
        MinAmpPow_Plots.append(Y_MIN)
        MaxAmpPow_Plots.append(Y_MAX)
        PLOTS["AMP-POWER"][1][p].set_yscale("log", subs=None)

    PLOTS["AMP-POWER"][1][Nplots-1].set_xscale("log", subs=None)

    Y_MIN, Y_MAX = min(MinAmpPow_Plots), max(MaxAmpPow_Plots)
    for p in range(Nplots):
        # Desactivate UserWarning, expected output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PLOTS["AMP-POWER"][1][p].set_ylim(Y_MIN, Y_MAX)

    # Fixing position of legends box outside the subplots
    PLOTS["AMP-POWER"][0].legend(loc="center right", bbox_to_anchor=(1, 0.5),
                                 fancybox=True, shadow=True)

    SaveAllRegionPlot("AmplitudePowerRegions", PLOTS["AMP-POWER"][0])

    print("finished!")
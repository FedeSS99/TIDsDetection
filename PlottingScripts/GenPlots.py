# Stations and Regions plotting routines
import PlottingScripts.StationsAndRegions.StatReg_Plots as StatRegPlot

# All Regions plotting routines
import PlottingScripts.AllRegions.Reg_Plots as RegPlot

from matplotlib.pyplot import close

# Start creating plots for either Region and Station to visualize the statistics given the data from DataDict


def StartAnalysisForStationsAndRegions(RegionName, StationName, Stat_or_Reg, DataDict, CMAP, NORM):
    if Stat_or_Reg == "Stat":
        PlotsResults = StatRegPlot.CreateFiguresForStationsAndRegions(
            StationName, Stat_or_Reg)
        print(f"Working on {StationName} Station...", end="\t")

    elif Stat_or_Reg == "Reg":
        PlotsResults = StatRegPlot.CreateFiguresForStationsAndRegions(
            RegionName, Stat_or_Reg)
        print(f"Working on {RegionName} Region...", end="\t")

    StatRegPlot.Add_TimeMonthsHistogramToPlot(DataDict["OCURRENCE"], CMAP, NORM, PlotsResults["OCURR"],
                                              RegionName, StationName, Stat_or_Reg)

    StatRegPlot.Add_PeriodHistogramToPlot(DataDict["PERIOD"], DataDict["TIME"], DataDict["MONTH"],
                                          PlotsResults["PERIOD"], RegionName, StationName, Stat_or_Reg)

    StatRegPlot.Add_BarsFreq_Month(DataDict["TIME"], DataDict["MONTH"],
                                   PlotsResults["DAY-NIGHT_BARS"], RegionName, StationName, Stat_or_Reg)

    for i in range(1, 4):
        close(i)

    print("finished!")


def StartAnalysisForOnlyRegion(DataDict, Region, RegionsInfo, CMAP, NORM, PLOTS, ListBoxPlots, ListLegendsBoxPlots):

    RegPlot.Add_TimeMonthsHistogramToPlot(DataDict["OCURRENCE"], CMAP, NORM, PLOTS["OCURR"],
                                          RegionsInfo[Region][2], len(RegionsInfo), Region)

    RegPlot.Add_PeriodHistogramToPlot(DataDict["PERIOD"], DataDict["TIME"], DataDict["MONTH"],
                                      PLOTS["PERIOD"], RegionsInfo[Region][2], Region)

    RegPlot.Add_BarsFreq_Month(DataDict["TIME"], DataDict["MONTH"], PLOTS["DAY-NIGHT_BARS"], 
                               RegionsInfo[Region][2], Region)
    
    # Add boxplots for time-power data in a figure for all regions
    BoxPlotObject = RegPlot.Add_TimePowerDataResultsToPlot(DataDict["TIME"], DataDict["POWER"],
                                                       PLOTS["POWER_VAR"], RegionsInfo[Region][0], 
                                                       RegionsInfo[Region][2])
    ListBoxPlots.append(BoxPlotObject)
    ListLegendsBoxPlots.append(Region)

    ## Add boxplots for time-amplitude data in a figure for all regions
    RegPlot.Add_AmplitudesAnalysis(DataDict["AVE_AMP"], DataDict["TIME"], 
                               DataDict["MONTH"], PLOTS["AMP_VAR"], RegionsInfo[Region][2], Region)

    # Add scatter plots of amplitude-power data in a figure for all regions
    RegPlot.Add_AmplitudePowerScatterPlot(DataDict["AVE_AMP"], DataDict["POWER"],
                                  DataDict["TIME"], DataDict["MONTH"], PLOTS["AMP_POWER"],
                                  RegionsInfo[Region][1], RegionsInfo[Region][2], Region)




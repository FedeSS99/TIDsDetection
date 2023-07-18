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


def StartAnalysisForOnlyRegion(DataDict, Region, RegionsInfo, CMAP, NORM, PLOTS):
    Nplots = len(RegionsInfo)
    Index = RegionsInfo[Region][2]

    print(f"Adding analysis results for {Region} region...", end="")
    # Add ocurrence map in local time and months in a figure for all regions
    RegPlot.Add_TimeMonthsHistogramToPlot(DataDict["OCURRENCE"], CMAP, NORM, PLOTS["OCURR"],
                                          Index, Nplots, Region)

    # Add period distribution in a figure for all regions
    RegPlot.Add_PeriodHistogramToPlot(DataDict["PERIOD"], DataDict["TIME"], DataDict["MONTH"],
                                      PLOTS["PERIOD"], Index, Region)

    # Add bars for total of events in day and night in a figure for all regions
    RegPlot.Add_BarsFreq_Month(DataDict["TIME"], DataDict["MONTH"], PLOTS["DAY-NIGHT_BARS"], 
                               Index, Nplots, Region)

    # Add analysis for time-amplitude data in a figure for all regions
    RegPlot.Add_QuantityVarAnalysis(DataDict["AVE_AMP"], DataDict["TIME"], DataDict["MONTH"],
                                   PLOTS["AMP_VAR"], Index, Region)
    
    # Add analysis for time-amplitude data in a figure for all regions
    RegPlot.Add_QuantityVarAnalysis(DataDict["POWER"], DataDict["TIME"], DataDict["MONTH"],
                                   PLOTS["POWER_VAR"], Index, Region)

    # Add scatter plots of amplitude-power data in a figure for all regions
    RegPlot.Add_AmplitudePowerScatterPlot(DataDict["AVE_AMP"], DataDict["POWER"],
                                          DataDict["TIME"], DataDict["MONTH"], PLOTS["AMP_POWER"],
                                          RegionsInfo[Region][1], Index, Region)
    print("finished!")

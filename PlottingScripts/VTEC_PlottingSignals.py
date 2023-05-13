from matplotlib.pyplot import figure, show, cm
from numpy import linspace, arange

def CreateSignalPlotFigure():
    timeTicks = arange(0, 25, 6)

    #Create main plotting figure to use for every prn number
    MainFigure = figure(1, figsize=(10, 5))
    SubFigureOrigSignalsCMN = MainFigure.add_subplot(211)
    SubFigureFiltSignalsCMN = MainFigure.add_subplot(212)
    MainFigure.subplots_adjust(hspace=0.0)

    #Setting subfigures labels and limits
    SubFigureOrigSignalsCMN.set_ylabel("VTEC")
    SubFigureOrigSignalsCMN.set_xlabel("Universal Time (Hours)")
    SubFigureOrigSignalsCMN.set_xlim(0.0, 24.0)
    SubFigureOrigSignalsCMN.axes.get_xaxis().set_ticks([])

    SubFigureFiltSignalsCMN.set_ylabel("dTEC")
    SubFigureFiltSignalsCMN.set_xlabel("Universal Time (Hours)")
    SubFigureFiltSignalsCMN.set_xlim(0.0, 24.0)
    SubFigureFiltSignalsCMN.set_xticks(timeTicks)

    return MainFigure, (SubFigureOrigSignalsCMN, SubFigureFiltSignalsCMN)


def CMN_SignalPlots(OrigTime,OrigVtec,FiltTime,FiltVtec, SignalsPlot):
    #Getting prn numbers to plot only the satellites in the given dataset
    prnNumbersOrig = OrigTime.keys()
    prnNumbersFilt = FiltTime.keys()

    #Create CMN_colors dictionary to save rgb colors for each prn Number plot
    CMN_Orig_colors, CMN_Filter_colors = dict(), dict()
    for prn,rgb in zip(prnNumbersOrig,cm.jet(linspace(0, 1, len(prnNumbersOrig)))):
        CMN_Orig_colors[prn] = rgb
    for prn,rgb in zip(prnNumbersFilt,cm.jet(linspace(0, 1, len(prnNumbersFilt)))):
        CMN_Filter_colors[prn] = rgb

    for prn in prnNumbersFilt:
        for plot in range(len(OrigTime[prn])):
            timeDataPRN = OrigTime[prn][plot]
            VTECDataPRN = OrigVtec[prn][plot]

            SignalsPlot[1][0].plot(timeDataPRN, VTECDataPRN,
            linewidth=1, color=CMN_Filter_colors[prn])

    for prn in prnNumbersFilt:
        for plot in range(len(FiltTime[prn])):
            timeDataPRN = FiltTime[prn][plot]
            VTECDataPRN = FiltVtec[prn][plot]
            if plot == 0:
                SignalsPlot[1][1].plot(timeDataPRN, VTECDataPRN,
                linewidth=1, color=CMN_Filter_colors[prn], label=f"{prn[:-2]}")
            else:
                SignalsPlot[1][1].plot(timeDataPRN, VTECDataPRN,
                linewidth=1, color=CMN_Filter_colors[prn])

    SignalsPlot[0].legend(loc="upper center",  mode="expand",
    fontsize=8, ncol=len(prnNumbersFilt)//2, borderaxespad = 0)
    #SignalsPlot[0].tight_layout()
    show()
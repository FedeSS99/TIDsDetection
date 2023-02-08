from tqdm import tqdm
from tkinter import Button, Tk
from os import listdir
import tkfilebrowser
from numpy import where, concatenate
from matplotlib import rcParams
from matplotlib.pyplot import close, style

from DataAnalysisRoutines.GetDataFile import SingleTIDs_Analysis
from DataAnalysisRoutines.HistogramOcurrence import Time_Months_Ocurrence_Analysis
from PlottingResultsRoutines.CreatePlots import *

#Ignore events that occoured in dates where a geomagnetic
#storm had a major effect in the Dst value
with open("tormentas-2018-2021.txt", "r") as StormDaysData:
    StormDays = []
    Lines = StormDaysData.readlines()

    for Line in Lines[1:]:
        Date = Line.split()[0]
        if Date not in StormDays:
            StormDays.append(Line.split()[0])

#Create list LIST_DIRS to save multiple directories
LIST_DIRS = []
def Select_Years_Directory():
    global LIST_DIRS
    LIST_DIRS.append([])
    LIST_DIRS[-1].append(tkfilebrowser.askopendirnames())

def StarAnnualAnalysis():
    #Stablish global list LIST_DIRS
    global LIST_DIRS

    #Create RESULTS list to save data of time, period and power from TIDs
    #saved with VTEC_MainRoutine_IndividualCMN.py
    RESULTS = []
    PowerPlot = CreateResultsFigurePower()
    for dirs in LIST_DIRS:
        if len(dirs[0]) > 0:
            ActiveDays = 0
            TotalDays = 0
            #Obtain the full path of the files located in the selected directories and
            #the dates and months of these same files
            for path in dirs[0]:
                files_month_full_path = [path+"/"+file for file in listdir(path) if file.endswith("_TIDs.dat")]
                Dates_TIDs = [fileName.split(".")[0].split("/")[-1][-15:-5] for fileName in files_month_full_path]
                MonthPerFile = [int(fileName.split("/")[-1].split("-")[2]) for fileName in files_month_full_path]

                #Get the name for the analysis' plots results
                NameOut = path.split("/")[-2]
                print(f"--Obtaining results of {path}--")
                TotalDays += len(files_month_full_path)
                
                ResultsTimeTID = []
                ResultsPeriodTID = []
                ResultsPowerTID = []
                MonthArray = []

                for fileTID, MonthFile, Date_TID in tqdm(zip(files_month_full_path, MonthPerFile, Dates_TIDs)):
                    if Date_TID not in StormDays:
                        Results = SingleTIDs_Analysis(fileTID)
                        SizeResults = Results[0].size
                        if SizeResults > 0:
                            ActiveDays += 1
                            MonthArray.append(SizeResults*[MonthFile])
                            ResultsTimeTID.append(Results[3])
                            ResultsPeriodTID.append(Results[5])
                            ResultsPowerTID.append(Results[6])

            MonthArray = concatenate(tuple(MonthArray), dtype=int)
            ResultsTimeTID = concatenate(tuple(ResultsTimeTID))
            #Get the timezone given NameOut
            if "NorthWest" in NameOut:
                TimeZone = -8.0
            elif "NorthEasth" in NameOut or "SouthWest" in NameOut:
                TimeZone = -6.0
            elif "SouthEast" in NameOut:
                TimeZone = -5.0
            #Apply timezone to get correct Local Time Hours
            ResultsTimeTID += TimeZone
            ResultsTimeTID = where(ResultsTimeTID < 0, ResultsTimeTID + 24.0, ResultsTimeTID)

            ResultsPeriodTID = concatenate(tuple(ResultsPeriodTID))
            ResultsPowerTID = concatenate(tuple(ResultsPowerTID))
            NumTIDs = ResultsTimeTID.size

            #Get histogram of TIDs' ocurrence and save all the data from the respective directory
            #in RESULTS
            HistogramOcurrence = Time_Months_Ocurrence_Analysis(ResultsTimeTID, MonthArray)
            RESULTS.append([ResultsTimeTID, MonthArray, HistogramOcurrence, ResultsPeriodTID, ResultsPowerTID, NameOut])

            print(f"# of TIDs:{NumTIDs}\nActive Days:{ActiveDays} Total Days:{TotalDays}\n")

    #Obtain the globam minimum and maximum of the ocurrence arrays of all the directories' data
    MIN, MAX = min([DataResults[2].min() for DataResults in RESULTS]), max([DataResults[2].max() for DataResults in RESULTS])
    for DataResults in RESULTS:
        #Get a string Coord to use in the analysis' plots results
        NamePlot = DataResults[-1]
        if "NorthEasth" in NamePlot:
            Coord = "(25,25°, -99.9°)"
            NamePower = "LNIG-MNIG"
            ColorPower = "red"
            RegIndex = 0
        elif "SouthWest" in NamePlot:
            Coord = "(19,81°, -101,69°)"
            NamePower = "UCOE"
            ColorPower = "green"
            RegIndex = 1
        elif "NorthWest" in NamePlot:
            Coord = "(20,15°, -87,4°)"
            NamePower = "PTEX"
            ColorPower = "blue"
            RegIndex = 2
        elif "SouthEast" in NamePlot:
            Coord = "(32,28°, -116,52°)"
            NamePower = "CN24-TGMX"
            ColorPower = "black"
            RegIndex = 3

        #Start creating Matplotlib plot to visualize the statistics given the data from RESULTS
        PlotsResults = CreateFiguresResults(Coord)
        AddTimeMonthsHistogramToPlot(DataResults[2], MIN, MAX, PlotsResults, NamePlot)
        AddPeriodHistogramToPlot(DataResults[3], PlotsResults, NamePlot)
        addTimePowerDataResultsToPlot(DataResults[0], DataResults[4], PowerPlot, ColorPower, NamePower, RegIndex)
        BarsFreq_Month(DataResults[0], DataResults[1], PlotsResults, NamePlot)
        for i in range(2,5):
            close(i)
    
    PowerPlot[1].set_yscale("log", subs=None)
    PowerPlot[0].legend(loc=1)
    PowerPlot[0].savefig(f"./../Resultados/PowerDistributionStations.png")
    close(1)
    #Clear LIST_DIRS for next set of directories
    LIST_DIRS.clear()


if __name__=="__main__":
    # Setting plotting format for all figures
    rcParams["font.family"] = "serif"
    rcParams["savefig.dpi"] = 400

    #Create Tkinter app to select and extract data from TIDs
    #identified in time series of dTEC given .Cmn files
    window = Tk()
    window.geometry('360x100')
    window.resizable(width=False, height=False)
    window.title('Annual TIDs Data Analysis')

    ButtonSelect = Button(window, text='Select Years', command=lambda: Select_Years_Directory())
    ButtonStarAnalysis = Button(window, text="Start Analysis", command=lambda: StarAnnualAnalysis())
    ButtonClose = Button(window, text='Close window', command=lambda: window.quit())
    ButtonSelect.pack()
    ButtonStarAnalysis.pack()
    ButtonClose.pack()
    window.mainloop()

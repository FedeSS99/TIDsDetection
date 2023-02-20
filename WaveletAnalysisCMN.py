from tkinter.filedialog import askopenfile
from tkinter import Button, Tk
from matplotlib import use
from matplotlib.pyplot import rcParams
import os

from numpy import array, where
from collections import Counter

from DataScripts.VTEC_IntervalsSubRoutines import ObtainIntervalsWith_SGFilter
from DataScripts.VTEC_WaveletSpectrum import CMN_WaveletAnalysis
from PlottingScripts.VTEC_PlottingSignals import CMN_SignalPlots, CreateSignalPlotFigure

def select_file(window):
    window.cmn_file = askopenfile(title="Select cmn file to read", filetypes=[("Cmn", "*.Cmn")])
    #If the status of window.cmn_file
    #doesn´t change, dont do anything
    if window.cmn_file is None:
        pass

    else:
        os.system('cls' if os.name == 'nt' else 'clear')
        cmn_file_path = window.cmn_file.name
        new_path_name = cmn_file_path.split("/")[-1]
        plot_name = new_path_name.split(".")[0]
        STATION_NAME = plot_name[:4]
        SAVEFILE_PATH = "../Análisis/"+STATION_NAME

        if not os.path.exists(SAVEFILE_PATH):
            os.mkdir(SAVEFILE_PATH)

        #------------------------------------------------------------------------------------
        with open(cmn_file_path, "+r") as cmn_file:
            cmn_file_lines = cmn_file.readlines()[5:]

            #After reading the whole .cmn file, it is needed to have saved only the
            #data that corresponds to a elevation greater than 30.0 degrees
            elevation = array([float(line.split()[4]) for line in cmn_file_lines])
            elevation_filter = where(elevation>=30.0, True, False)

            time_cmn = array([float(line.split()[1]) for line in cmn_file_lines])[elevation_filter]
            fixed_time_cmn = where(time_cmn>=0.0, time_cmn, time_cmn+abs(time_cmn))
            prn_cmn = array([float(line.split()[2]) for line in cmn_file_lines])[elevation_filter]
            vtec_cmn = array([float(line.split()[8]) for line in cmn_file_lines])[elevation_filter]

        #Then each read line is saved in different arrays on the condition
        #of being from the same prn
        prn_values = [prn_value for prn_value, count in Counter(prn_cmn).items() if count>1]
        cmn_time_vtec_readings = dict()
        for prn_value in prn_values:
            prn_filter = where( prn_cmn==prn_value, True, False)
            cmn_time_vtec_readings[str(prn_value)] = [fixed_time_cmn[prn_filter],vtec_cmn[prn_filter]]

        #------------------------------------------------------------------------------------
        #Define the value of dj to evaluate the max magnitud of the scales to apply
        #on the Wavelet transforms and the time window to obtain the window sizes of all
        #the subintervals on each satellite
        dj = 0.025
        time_window = 1.0

        #Getting dictionaries for time-vtec data without tendency using Savitzky-Golay filter
        Time_cmn, Vtec_cmn, TimeFilter_cmn, VtecFilter_cmn = ObtainIntervalsWith_SGFilter(cmn_time_vtec_readings, time_window)

        # Create and show original and detrended signals on plots
        SignalPlots = CreateSignalPlotFigure(plot_name)
        CMN_SignalPlots(Time_cmn, Vtec_cmn, TimeFilter_cmn, VtecFilter_cmn, SignalPlots)
        
        # Start a file to save TIDs data from detrended signals
        with open(SAVEFILE_PATH+"/"+plot_name+"_TIDs.dat", "w") as OutTIDs: 
            OutTIDs.write(f"#TIDs data obtained with {plot_name}\n")
            OutTIDs.write("#TimeTID PeriodTID PowerTID InitTime FinalTime InitPeriod FinalPeriod minSignal maxSignal\n")
            CMN_WaveletAnalysis(TimeFilter_cmn, VtecFilter_cmn, dj, plot_name, OutTIDs)


if __name__=="__main__":
    #Change Matplotlib's backend and font family
    use('TkAgg')
    rcParams.update({'font.family':'serif'})
    rcParams.update({'savefig.dpi': 300})

    #Create and show Tkinter app for user to start the analysis of data
    #in .Cmn file
    window = Tk()
    window.geometry('360x100')
    window.resizable(width=False, height=False)
    window.title("Analyze .Cmn file's VTEC data")

    boton = Button(window, text='Select a .Cmn file', command=lambda: select_file(window))
    boton_cerrar = Button(window, text='Close window', command=lambda: window.quit())
    boton.pack()
    boton_cerrar.pack()
    window.mainloop()

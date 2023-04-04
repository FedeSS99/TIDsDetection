from tkinter.filedialog import askopenfile
from tkinter import Button, Tk
from matplotlib import use
from matplotlib.pyplot import rcParams
import os

from numpy import array, where
from collections import Counter

from DataScripts.VTEC_IntervalsSubRoutines import ObtainIntervalsWith_SGFilter
from PlottingScripts.VTEC_PlottingSignals import CMN_SignalPlots, CreateSignalPlotFigure

use('TkAgg')
rcParams["font.family"] = "serif"
rcParams["savefig.dpi"] = 400

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

        #------------------------------------------------------------------------------------
        with open(cmn_file_path, "+r") as cmn_file:
            cmn_data_lines = cmn_file.readlines()[5:]

            #After reading the whole .cmn file, it is needed to have saved only the
            #data that corresponds to a elevation greater than 30.0 degrees
            elevation = array([float(line.split()[4]) for line in cmn_data_lines])
            elevation_filter = where( elevation>=30.0, True, False)

            time_cmn = array([float(line.split()[1]) for line in cmn_data_lines])[elevation_filter]
            fixed_time_cmn = where(time_cmn>=0.0, time_cmn, time_cmn+abs(time_cmn))
            prn_cmn = array([float(line.split()[2]) for line in cmn_data_lines])[elevation_filter]
            vtec_cmn = array([float(line.split()[8]) for line in cmn_data_lines])[elevation_filter]

        #Then each read line is saved in different arrays on the condition
        #of being from the same prn
        prn_values = [prn_value for prn_value, count in Counter(prn_cmn).items() if count>1]
        cmn_time_vtec_readings = dict()
        for prn_value in prn_values:
            prn_filter = where( prn_cmn==prn_value, True, False)
            cmn_time_vtec_readings[str(prn_value)] = [fixed_time_cmn[prn_filter],vtec_cmn[prn_filter]]

        #------------------------------------------------------------------------------------
        #Define the time window to obtain the window sizes of all the subintervals on each satellite
        time_window = 1.0

        #Getting dictionaries for time-vtec data without tendency using running average on .cmn file
        Time_cmn, Vtec_cmn, TimeFilter_cmn, VtecFilter_cmn = ObtainIntervalsWith_SGFilter(cmn_time_vtec_readings, time_window)

        SignalPlots = CreateSignalPlotFigure(plot_name)
        CMN_SignalPlots(Time_cmn, Vtec_cmn, TimeFilter_cmn, VtecFilter_cmn, SignalPlots)

if __name__=="__main__":
    window = Tk()
    window.geometry('360x100')
    window.resizable(width=False, height=False)
    window.title("Visualize .Cmn file's VTEC data")

    boton = Button(window, text='Select a .Cmn file', command=lambda: select_file(window))
    boton_cerrar = Button(window, text='Close window', command=lambda: window.quit())
    boton.pack()
    boton_cerrar.pack()
    window.mainloop()

from numpy import array, split, where, diff
from scipy.signal import savgol_filter


def FindOptimalOrder_SGF(DataInterval, WindowSize, PRN, NumInt):
    #Pre computed values for optimar order and error for the filter
    #are written, expecting to be changed through the next loop
    OptimalR2 = 0.0
    OptimalOrder = 1
    IntervalTime = diff(DataInterval[PRN][0][NumInt]).mean()
    # Only explore polynomial orders from 1 to 10
    for order in range(1, 11):
        interval_tendency = savgol_filter(DataInterval[PRN][1][NumInt], WindowSize[PRN][NumInt],
        order, delta=IntervalTime, mode="nearest")

        meanOriSignal = DataInterval[PRN][1][NumInt].mean()
        sr = sum((DataInterval[PRN][1][NumInt] - interval_tendency)**2.0)
        st = sum((DataInterval[PRN][1][NumInt] - meanOriSignal)**2.0)

        # Computing R2 score
        R2 = 1.0 - (sr/st)
        if OptimalR2 < R2 < 1.0:
            OptimalR2  = R2
            OptimalOrder = order

    return OptimalOrder

#--------------------------------------------------------------------------------------
def ObtainIntervalsWith_SGFilter(time_vtec_readings, time_window):
    #Save all the future obtained intervals on the whole data set in all_intervals
    #and saved the corresponding window size for each one of these
    all_intervals = dict()
    all_windows_sizes = dict()

    #The values of time of each subinterval of each satellite data set is saved in
    #time_no_tendency and the given VTEC values of these same subintervals in
    #vtec_no_tendency
    time_no_tendency = dict()
    vtec_no_tendency = dict()
    #Also, save original time-vtec data
    timeOriginal = dict()
    vtecOriginal = dict()

    #A for cycle that will split and save each satellite data set in the previous
    #list
    prnNumbers =time_vtec_readings.keys()
    for prn in prnNumbers:
        time_difference = diff(time_vtec_readings[prn][0])
        minDiff = time_difference.min()
        intervals_cut = where(time_difference>2.0*minDiff)[0]

        if intervals_cut.shape[0] > 0:
            dataIntervals = (split(time_vtec_readings[prn][0], intervals_cut+1), split(time_vtec_readings[prn][1], intervals_cut+1))
            all_intervals[prn] = [dataIntervals[0], dataIntervals[1]]
        else:
            all_intervals[prn] = [array([time_vtec_readings[prn][0]]), array([time_vtec_readings[prn][1]])]

        timeOriginal[prn] = all_intervals[prn][0]
        vtecOriginal[prn] = all_intervals[prn][1]

        #The average time interval for each time data set is required to know
        #the windows size that will be used on the running average
        windows_sizes = []
        for time_interval in all_intervals[prn][0]:
            average_delta_t = diff(time_interval).mean()
            windows_sizes.append(int(time_window/average_delta_t + 0.5))

        all_windows_sizes[prn] = windows_sizes


    #Getting detrended data for each subinterval for each PRN number with the Savitzky-Golay filter
    for prn in prnNumbers:
        time_interval_no_tendency = []
        vtec_interval_no_tendency = []
        for p in range(len(all_intervals[prn][1])):
            if all_windows_sizes[prn][p]<len(all_intervals[prn][1][p]):
                # Find order of SG filter that minimizes error
                OptimalOrder = FindOptimalOrder_SGF(all_intervals, all_windows_sizes, prn, p)

                #Detrending data with the optimal filter order with the minimum error
                interval_tendency = savgol_filter(all_intervals[prn][1][p], all_windows_sizes[prn][p],
                OptimalOrder,delta=diff(all_intervals[prn][0][p]).mean(),mode="nearest")

                sizeIntervalTendency = interval_tendency.shape[0]
                num_elements_interval = (all_intervals[prn][1][p]).shape[0]

                time_interval_no_tendency.append(array(all_intervals[prn][0][p][num_elements_interval-sizeIntervalTendency:]))
                vtec_interval_no_tendency.append(array(all_intervals[prn][1][p][num_elements_interval-sizeIntervalTendency:]-interval_tendency))

        time_no_tendency[prn] = time_interval_no_tendency
        vtec_no_tendency[prn] = vtec_interval_no_tendency

    return timeOriginal, vtecOriginal, time_no_tendency, vtec_no_tendency

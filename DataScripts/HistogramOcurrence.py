from numpy import histogram, zeros, float32

"""
Function that returns a 2D array with a total of 288 bins
(24 hours x 12 months) with purpose to obtain the pecentage 
of TIDs that occur given a month and hour range by the following
condition:

Ocurr[Month,Hour] = 100 * (Amount of TIDs in the given mounth and range [hour, hour+1])
                          -------------------------------------------------------------
                                        Amount of TIDs in the whole given month
"""

def GetOcurrenceArray(Time, Months_TIDs):
    # Setting number of bins and time range for histogram
    TimeBins = 24
    TimeRange = (0.0, 24.0)
    MonthBins = 12

    # Numpy array to save the histogram data for each month per year
    HistogramMonths = zeros((MonthBins, TimeBins), dtype=float32)
    setMonths = set(Months_TIDs)

    for Month in setMonths:
        MaskMonths = Months_TIDs == Month
        TimeByMonth = Time[MaskMonths]
        index = Month - 1

        TimeHistogramByMonth, _ = histogram(TimeByMonth, bins=TimeBins, range=TimeRange)
        HistogramMonths[index, :] = 100.0*(TimeHistogramByMonth[:]/TimeHistogramByMonth.sum())


    return HistogramMonths

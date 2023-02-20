from numpy import histogram, zeros, float32


def Time_Months_Ocurrence_Analysis(Time, Months_TIDs):
    #Setting number of bins and time range for histogram
    TimeBins = 24
    TimeRange = (0.0, 24.0)
    MonthBins = 12

    #Numpy array to save the histogram data for each month per year
    HistogramMonths = zeros((MonthBins,TimeBins), dtype=float32)
    setMonths = set(Months_TIDs)

    for Month in setMonths:
        MaskMonths = Months_TIDs==Month
        TimeByMonth = Time[MaskMonths]
        index = Month - 1

        TimeHistogramByMonth, _ = histogram(TimeByMonth, bins=TimeBins, range=TimeRange)
        TimeHistogramByMonth = 100*TimeHistogramByMonth/TimeHistogramByMonth.sum()
        HistogramMonths[index,:] = TimeHistogramByMonth[:]

    #HistogramMonths *= (100/HistogramMonths.sum())
    return HistogramMonths
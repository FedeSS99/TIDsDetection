from numpy import histogram, zeros, float32
from scipy.stats import iqr

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
        if SumMonthHistogram := TimeHistogramByMonth.sum():
            HistogramMonths[index, :] = 100.0*(TimeHistogramByMonth[:]/SumMonthHistogram)
        else:
            HistogramMonths[index, :] = 0.0


    return HistogramMonths

def GetPowerTimeArray(Power_TIDs, Time_TIDs, Months_TIDs):
    # Setting number of bins and time range for histogram
    TimeBins = 24
    TimeRange = (0.0, 24.0)
    MonthBins = 12

    # Numpy array to save the histogram data for each month per year
    PowerArray = zeros((MonthBins, TimeBins), dtype=float32)
    setMonths = set(Months_TIDs)

    for Month in setMonths:
        MonthArray = (Months_TIDs == Month)
        PowerValuesMonth = Power_TIDs[MonthArray]
        for InitTime in range(24):
            TimeInterval = (InitTime <= Time_TIDs[MonthArray]) & ( Time_TIDs[MonthArray] < InitTime + 1.0)
            AcceptPowerValues = PowerValuesMonth[TimeInterval]
            if AcceptPowerValues.size:
                PowerArray[Month-1, InitTime] = AcceptPowerValues.max()

    return PowerArray
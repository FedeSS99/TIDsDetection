from numpy import float64, array

"""
Function that read and returns all the TIDs' data from the given
input file name (TID_path); at first everything is saved in a list
but later turned into a numpy array for easier use in the main file
(TIDs_StatsAnalysis.py).
"""

def GetSingleTID_Data(TID_path):
    #------------------------------------------------------------------------------------
    #Reading TIDs events from TID_path
    with open(TID_path, "+r") as tid_file:
        tid_file_lines = tid_file.readlines()

        #Reading and saving data from tid_file
        TimeTIDS, PeriodTIDS, PowerTIDS = [], [], []
        MinAmps, MaxAmps = [], []
        #Starting at third line
        for line in tid_file_lines[2:]:
            TID_Data = [float(x) for x in line.split()] 

            TimeTIDS.append( TID_Data[0] )
            PeriodTIDS.append( TID_Data[1] )
            PowerTIDS.append( TID_Data[2] )
            MinAmps.append( TID_Data[7])
            MaxAmps.append( TID_Data[8])

        #Declaring lists of data as numpy arrays
        TimeTIDS = array(TimeTIDS, dtype=float64)
        PeriodTIDS = array(PeriodTIDS, dtype=float64)
        PowerTIDS = array(PowerTIDS, dtype=float64)
        MinAmps = array(MinAmps, dtype=float64)
        MaxAmps = array(MaxAmps, dtype=float64)

    if TimeTIDS.size == PeriodTIDS.size == PowerTIDS.size:
        return {"TIME":TimeTIDS, "PERIOD":PeriodTIDS, "POWER":PowerTIDS,
                 "MIN_AMPS":MinAmps, "MAX_AMPS":MaxAmps}
    else:
        return False
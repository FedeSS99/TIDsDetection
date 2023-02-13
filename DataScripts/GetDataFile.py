from numpy import float64, array

def SingleTIDs_Analysis(TID_path):
    #------------------------------------------------------------------------------------
    #Reading TIDs events from TID_path
    with open(TID_path, "+r") as tid_file:
        tid_file_lines = tid_file.readlines()

        #Reading and saving data from tid_file
        TimeTIDS, PeriodTIDS, PowerTIDS = [], [], []
        #Starting at third line
        for line in tid_file_lines[2:]:
            TID_Data = [float(x) for x in line.split()] 

            TimeTIDS.append( TID_Data[0] )
            PeriodTIDS.append( TID_Data[1] )
            PowerTIDS.append( TID_Data[2] )

        #Declaring lists of data as numpy arrays
        TimeTIDS = array(TimeTIDS, dtype=float64)
        PeriodTIDS = array(TimeTIDS, dtype=float64)
        PowerTIDS = array(TimeTIDS, dtype=float64)

    if TimeTIDS.size == PeriodTIDS.size == PowerTIDS.size:
        return TimeTIDS, PeriodTIDS, PowerTIDS
    else:
        return False
from pandas import read_excel

NameOut = "TerminatorHours_NW.dat"
Data = read_excel("NOAA_Solar_Calculations_year.xlsx", usecols="Y:Z")
Data.reset_index()

TiempoFloat = lambda TiempoS: float(TiempoS.hour) + float(TiempoS.minute)/60.0 + float(TiempoS.second)/3600.0

for index in Data.index:
    Data["Sunrise Time (LST)"][index] = TiempoFloat(Data["Sunrise Time (LST)"][index])
    Data["Sunset Time (LST)"][index] = TiempoFloat(Data["Sunset Time (LST)"][index]) 

Data.to_csv(NameOut, sep=" ")
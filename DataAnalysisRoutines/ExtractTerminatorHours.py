from pandas import ExcelFile

# Reading NOAA calculations data
file = "NOAA_Solar_Calculations_year.xlsx"
data = ExcelFile(file)

# Only parsin data related to solar terminators
df = data.parse("Calculations")
strSunset, strSunrise = "Sunset Time (LST)", "Sunrise Time (LST)"

# Getting lists for terminators data
SunriseTime = df[strSunrise].tolist()
SunsetTime = df[strSunset].tolist()

# Lambda function to apply
ToNum = lambda dateTime: dateTime.hour + dateTime.minute/60.0 + dateTime.second/3600.0
SunriseTime = list(map(ToNum, SunriseTime))
SunsetTime = list(map(ToNum, SunsetTime))

with open("TerminatorHours.dat", "w") as OutFile:
    if len(SunriseTime) == len(SunsetTime):
        OutFile.write("Sunrise Time (LST) Sunset Time (LST)\n")
        for dateRise, dateSet in zip(SunriseTime, SunsetTime):
            OutFile.write(f"{dateRise} {dateSet}\n")
    else:
        print("Non existent or equal amount of data for each time sets")

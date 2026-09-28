import configparser
import numpy as np 
import pandas as pd


config = configparser.ConfigParser()
config.read('config.ini')

datafile = config['data']['file']


data = []


try:
    with open(datafile,'r') as datacsv:
        columns = datacsv.readline().strip().split(',')
        for d in datacsv:
            d = d.rstrip().split(',')
            while ( '' in d or "" in d):
                i =  d.index('')
                d[i] = None
            data.append(d)
except FileNotFoundError:
    print(datafile + " not found")
except PermissionError:
    print("Permission denied for " + datafile)

print(columns)
if (len(data) > 0):
    print(data[0])


df = pd.DataFrame(data, columns = columns)
df = df.astype({'model_year': 'int64','num_doors': 'float64', 'engine_displacement': 'float64', 'num_cylinders': 'float64', 'horsepower': 'float64', 'vehicle_weight': 'float64', 'acceleration': 'float64', 'fuel_efficiency_mpg': 'float64'})
print(df.dtypes)
print("Q1") 
print(pd.__version__)


print("Q2") 
print(df.shape)


print("Q3") 
print(df.fuel_type.nunique())

print("Q4")
nullSum = df.isnull().sum()
print((nullSum >0 ).sum())

print("Q5") 
print(df.fuel_efficiency_mpg.max())

print("Q6")
mean_horsepower = df.horsepower.median()
print(mean_horsepower)
print(df.groupby("horsepower").count().sort_values(by=['model_year'], ascending=False))
dffna = df.fillna(252)
print(dffna)
print(dffna.horsepower.median())


print("Q7")
print(df[df.origin == "Asia"][['model_year','vehicle_weight']][:7])
X = df[df.origin == "Asia"][['model_year','vehicle_weight']][:7].to_numpy()
print(X)
XTX = X.T.dot(X)
print(XTX)
XTXinv = np.linalg.inv(XTX)
y = [1100, 1300, 800, 900, 1000, 1100, 1200]
w = XTXinv.dot(X.T).dot(y)
print("")
print(w)
print(w.sum())


print("")
print(df)
##mean_horsepower = df.horsepower.median()
##print(mean_horsepower)
##print(df.groupby("horsepower").count().sort_values(by=['model_year'], ascending=False))
##dffna = df.fillna(252)
##print(dffna)
##print(dffna.horsepower.median())
###decreased
##print(df[df.origin == "Asia"][['model_year','vehicle_weight']][:7])
##X = df[df.origin == "Asia"][['model_year','vehicle_weight']][:7].to_numpy()
##print(X)
##XTX = X.T.dot(X)
##print(XTX)
##XTXinv = np.linalg.inv(XTX)
##y = [1100, 1300, 800, 900, 1000, 1100, 1200]
##w = XTXinv.dot(X.T).dot(y)
##print("")
##print(w)
##print(w.sum())

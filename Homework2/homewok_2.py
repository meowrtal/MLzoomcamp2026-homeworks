import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df= pd.read_csv('data.csv')

#EDA

print("First part of the data")
print(df.head())
print("columns types")
print(df.dtypes)

print("column details")
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())

sns.histplot(df.fuel_efficiency_mpg,bins=50)
plt.show()
print("")
#Q1
print("Q1")
print("")
print(df.isnull().sum())

print("The missing values are horsepower")


print("")
#Q2

print("Q2")
print("")
print(df.horsepower.median())


#data prep
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

print("")
#Q3
print("Q3")
print("")

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(len(df_train), len(df_val),len(df_test))

y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values

print(len(y_train), len(y_val),len(y_test))

del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

base = ['engine_displacement', 'horsepower', 
        'vehicle_weight','model_year']

def prepare_X(df,fill):
    df = df.copy()
    df_nonull = df[base].fillna(fill)
    X = df_nonull.values
    return X

def train_linear_regression(X,y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    XT = X.T
    w_full = XTX_inv.dot(XT).dot(y)

    return w_full[0],w_full[1:]

train_mean = df_train.horsepower.median()

X_train_0 = prepare_X(df_train,0)
X_train_mean = prepare_X(df_train,train_mean)

w0, w = train_linear_regression(X_train_0,y_train)
print(w0,w)
y_pred_0 = w0 + X_train_0.dot(w)


w_mean0, w_mean = train_linear_regression(X_train_mean,y_train)
print(w_mean0, w_mean)
y_pred_mean = w_mean0 + X_train_mean.dot(w_mean)

sns.histplot(y_pred_0, color= 'red', alpha=0.5, bins = 50)
sns.histplot(y_pred_mean, color= 'green', alpha=0.5, bins = 50)
sns.histplot(y_train, color = 'blue', alpha=0.5, bins = 50)

plt.show()

print("")


X_val_0 = prepare_X(df_val,0)

X_val_mean = prepare_X(df_val,train_mean)

y_pred_val_0 = w0 + X_val_0.dot(w)
y_pred_val_mean = w_mean0 + X_val_mean.dot(w_mean) 

sns.histplot(y_pred_val_0, color= 'red', alpha=0.5, bins = 50)
sns.histplot(y_pred_val_mean, color= 'green', alpha=0.5, bins = 50)
sns.histplot(y_val, color = 'blue', alpha=0.5, bins = 50)

plt.show()


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

rmse_0 = rmse(y_val, y_pred_val_0)
rmse_mean = rmse(y_val, y_pred_val_mean)

print(round(rmse_0, 3),round(rmse_mean, 3))


#Q04
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    X_train = prepare_X(df_train,0)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val,0)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)

    print(r,'\t', w0,'\t', round(score, 4))




#Q05

all_scores = []

for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    
    np.random.seed(s)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.fuel_efficiency_mpg.values
    y_val = df_val.fuel_efficiency_mpg.values
    y_test = df_test.fuel_efficiency_mpg.values

    del df_train['fuel_efficiency_mpg']
    del df_val['fuel_efficiency_mpg']
    del df_test['fuel_efficiency_mpg']

    X_train = prepare_X(df_train,0)
    w0, w = train_linear_regression(X_train,y_train)

    X_val = prepare_X(df_val,0)

    y_pred = w0 + X_val.dot(w)

    score = rmse(y_val,y_pred)

    all_scores.append(score)

print(all_scores)
print(len(all_scores))
print(round(np.std(all_scores),3))


#Q06

s = 9

df_full_train = pd.concat([df_train,df_val])
df_full_train = df_full_train.reset_index(drop=True)

X_full_train = prepare_X(df_full_train,0)

y_full_train = np.concatenate([y_train, y_val])

w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

X_test = prepare_X(df_test,0)

y_pred_test = w0 + X_test.dot(w)

score = rmse(y_test,y_pred_test)

print(round(score,3)) 





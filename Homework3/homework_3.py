#homework 03
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data.csv')
print(df)
print(df.head().T)
print(df.dtypes)
print(df.isnull().sum())


df.annual_income = df.annual_income.fillna(0)
df.lead_score = df.lead_score.fillna(0)

df.lead_source = df.lead_source.fillna('NA')
df.industry = df.industry.fillna('NA')
df.employment_status = df.employment_status.fillna('NA')
df.location = df.location.fillna('NA')

print(df)

#Q1

print(df.industry.value_counts())

numerical = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']
categorical = ['lead_source', 'industry', 'employment_status', 'location']

#Q2
print(df[numerical].corrwith(df.converted))

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)

print(len(df),len(df_full_train),len(df_train),len(df_val),len(df_test))

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)


y_train = df_train.converted.values
y_val = df_val.converted.values
y_test = df_test.converted.values

del df_train['converted']
del df_val['converted']
del df_test['converted']
del df_full_train['converted']

#Q03
def mutual_info_converted_score(series):
    return mutual_info_score(series, y_train)

mi = df_train[categorical].apply(mutual_info_converted_score)
print(mi.sort_values(ascending=False))

#Q04




train_dict = df_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

print(dv.get_feature_names_out())
print(X_train.shape)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

converted_decision = (y_pred >= 0.5)

df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = converted_decision.astype(int)
df_pred['actual'] = y_val


df_pred['correct'] = df_pred.prediction == df_pred.actual

print(df_pred)
print(round(df_pred.correct.mean(),2))

original_score = df_pred.correct.mean()

#Q05

features = df_train.columns.to_list()
eliminate = ['interaction_count',  'number_of_courses_viewed', 'lead_source']

scores = pd.DataFrame(columns=['eliminated_feature', 'accuracy', 'difference'])

for feature in eliminate:
    subset = features.copy()
    subset.remove(feature)
    print(subset)
    print(feature)
    

    dv = DictVectorizer(sparse=False)
    
    train_dict = df_train[subset].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train,y_train)

    val_dict = df_val[subset].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    y_pred = model.predict_proba(X_val)[:, 1]
    
    converted_decision = (y_pred >= 0.5)
    
    df_pred = pd.DataFrame()
    df_pred['probability'] = y_pred
    df_pred['prediction'] = converted_decision.astype(int)
    df_pred['actual'] = y_val
    df_pred['correct'] = df_pred.prediction == df_pred.actual

    score = df_pred.correct.mean()
    print(df_pred)
    print(original_score)
    print(score)
    
    scores.loc[len(scores)] = [feature, score, original_score - score]

print(scores)    


#Q06





C_list = [0.000001, 0.00001, 0.0001, 0.001]
scores = pd.DataFrame(columns=['C', 'accuracy'])
for c in C_list:
    
    train_dict = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]

    converted_decision = (y_pred >= 0.5)

    df_pred = pd.DataFrame()
    df_pred['probability'] = y_pred
    df_pred['prediction'] = converted_decision.astype(int)
    df_pred['actual'] = y_val


    df_pred['correct'] = df_pred.prediction == df_pred.actual


    score = df_pred.correct.mean()
    
    scores.loc[len(scores)] = [c, score]

print(scores)





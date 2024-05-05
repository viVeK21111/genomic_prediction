import pandas as pd
df=pd.read_csv("G:\My Drive\ps\data\ddata.csv") 
df=df.drop(columns=['Unnamed: 0'])
x=df.iloc[:,:-2]
x['height']=df['height']
y=df.iloc[:,-2]
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x, y)
z=X_resampled['height']
X_resampled=X_resampled.drop('height',axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_resampled,y_resampled,random_state=42,test_size=0.33) 
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
mm.fit(x_train)
x_train=mm.transform(x_train)
x_test=mm.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
model_r=RandomForestClassifier(n_estimators=15,max_depth=6,random_state=42,criterion="gini",max_features='sqrt',max_leaf_nodes=29) 
model_r.fit(x_train,y_train)
s_pred=model_r.predict(X_resampled)
s_pred=list(s_pred)
df_h=pd.DataFrame(X_resampled)
df_h['sub']=s_pred 
df_h['sub']=df_h['sub'].astype("category").cat.codes +1  
df_h['height']=z
df_h.columns=df_h.columns.astype(str)   
def Outlier_Treatment(df1,variable):
    global df_final
    percentile25=df1[variable].quantile(0.25)  #q1
    percentile75=df1[variable].quantile(0.75)  #q3
    iqr=percentile75-percentile25
    upper_limit=percentile75+1.5*iqr
    lower_limit=percentile25-1.5*iqr
    df_final = df1[ (df1[variable]>=lower_limit) & (df1[variable]<=upper_limit) ]
Outlier_Treatment(df_h,'height')
xh=df_final.iloc[:,:-1]
yh=df_final.iloc[:,-1] 
X_train,X_test,Y_train,Y_test=train_test_split(xh,yh,random_state=42,test_size=0.33)
from sklearn.ensemble import RandomForestRegressor
model_rr=RandomForestRegressor(n_estimators=14,max_depth=4,random_state=42,criterion='squared_error')
from sklearn.ensemble import BaggingRegressor
bg_model=BaggingRegressor(base_estimator=model_rr,n_estimators=15,random_state=42)
bg_model.fit(X_train,Y_train)

def lmc():
    return model_r
def lmr():
    return bg_model
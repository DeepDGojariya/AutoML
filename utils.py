import pandas as pd
import numpy as np
from pandas.core.arrays.sparse import dtype
from scipy.sparse.construct import random
import missingno as msgno
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import datetime
import re
import pickle
import joblib
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import confusion_matrix,classification_report,r2_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import precision_recall_fscore_support as score


categorical_var,target_var,numeric_var,unwanted_var = [],[],[],[]

def empty_contents():
    global categorical_var
    global target_var
    global numeric_var
    global unwanted_var
    categorical_var,target_var,numeric_var,unwanted_var = [],[],[],[]
        
    

def get_categorical():
    return categorical_var

def get_target():
    return target_var

def get_numeric():
    return numeric_var

def get_unwanted():
    return unwanted_var

def set_var(c,t,n,u):
    global categorical_var
    global target_var
    global numeric_var
    global unwanted_var
    categorical_var,target_var,numeric_var,unwanted_var = c,t,n,u

    
    
def missing_value_graph(filepath):
    df = pd.read_csv(filepath)
    fig = msgno.matrix(df)
    fig_copy = fig.get_figure()
    uniq_id = datetime.datetime.now().strftime("%c")
    uniq_id = ('').join(uniq_id.split())
    uniq_id = re.sub(":","",uniq_id)
    fig_path = "missing_{}.png".format(uniq_id) 
    fig_copy.savefig("static/{}".format(fig_path))
    return fig_path

def get_attributes(filepath):
    df = pd.read_csv(filepath)
    return [i for i in df.columns]
    

def get_numeric_attributes(filepath):
    df = pd.read_csv(filepath)
    lst=[]
    for i,j in zip(df.columns,df.dtypes):
        if j=='int64' or j=='float64':
            lst.append(i)
    return lst

def get_numeric_missing_attributes(filepath):
    df = pd.read_csv(filepath)
    lst=[]
    for i in numeric_var:
        if df[i].isnull().any():
            lst.append(i)
    return lst

def get_categorical_missing_attributes(filepath):
    df = pd.read_csv(filepath)
    lst=[]
    for i in categorical_var:
        if df[i].isnull().any():
            lst.append(i)
    return lst

def create_histogram(attr,bins,filepath):
    df = pd.read_csv(filepath)
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    fig_ = sns.histplot(df[attr],bins=int(bins),ax=ax)
    fig_copy = fig_.get_figure()
    uniq_id = datetime.datetime.now().strftime("%c")
    uniq_id = ('').join(uniq_id.split())
    uniq_id = re.sub(":","",uniq_id)
    fig_path = "histogram_{}.png".format(uniq_id) 
    fig_copy.savefig("static/{}".format(fig_path))
    return fig_path

def create_pairplot(filepath):
    df = pd.read_csv(filepath)
    fig_ = sns.pairplot(df)
    uniq_id = datetime.datetime.now().strftime("%c")
    uniq_id = ('').join(uniq_id.split())
    uniq_id = re.sub(":","",uniq_id)
    fig_path = "pairplot_{}.png".format(uniq_id)
    fig_.savefig("static/{}".format(fig_path))
    return fig_path

def create_scatterplot(attr1,attr2,filepath):
    df = pd.read_csv(filepath)
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    fig_ = sns.scatterplot(df[attr1],df[attr2],ax=ax)
    fig_copy = fig_.get_figure()
    uniq_id = datetime.datetime.now().strftime("%c")
    uniq_id = ('').join(uniq_id.split())
    uniq_id = re.sub(":","",uniq_id)
    fig_path = "scatterplot_{}.png".format(uniq_id)
    fig_copy.savefig("static/{}".format(fig_path))
    return fig_path
    
def create_barplot(attr1,attr2,filepath):
    df = pd.read_csv(filepath)
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    fig_ = sns.barplot(x=df[attr1],y=df[attr2],ax=ax)
    fig_copy = fig_.get_figure()
    uniq_id = datetime.datetime.now().strftime("%c")
    uniq_id = ('').join(uniq_id.split())
    uniq_id = re.sub(":","",uniq_id)
    fig_path = "barplot_{}.png".format(uniq_id)
    fig_copy.savefig("static/{}".format(fig_path))
    return fig_path
    
def update_csv_(attr_ui,filepath):
    global numeric_var
    print("Attr_ui= ",attr_ui)
    print("Hiii")
    df = pd.read_csv(filepath)
    for (a,b,c) in attr_ui:
        if b=='':
            if c=='Mean':
                mean_val = pd.to_numeric(df[a], errors='coerce').mean()
                df[a].fillna(round(mean_val,2),inplace=True)
            elif c=='Median':
                median_val = pd.to_numeric(df[a], errors='coerce').median()
                df[a].fillna(round(median_val,2),inplace=True)
            elif c=='Mode':
                df[a].fillna(df[a].mode()[0],inplace=True)
            elif c=='DR':
                df.dropna(axis=0,subset=[a],inplace=True)
            elif c=='DC':
                df.drop(axis=1,columns=[a],inplace=True)
        else:
            df[a].fillna(float(b),inplace=True)
    for attr in df.columns:
        if attr in numeric_var:
            for i,val in enumerate(df[attr]):
                try:
                    float(val)
                except:
                    df[attr].iloc[i]=np.nan

        if attr in unwanted_var:
            print(attr,"###################")
            df.drop(axis=1,columns=[attr],inplace=True)
    df.dropna(inplace=True)
    df.to_csv(filepath,index=False)

def encode_categorical(enc,filepath):
    pass

def train_model(filepath,algo,size):
    global categorical_var
    global target_var
    global unwanted_var
    df = pd.read_csv(filepath)
    for i in unwanted_var:
        try:
            df.drop(columns=[i],axis=1,inplace=True)
        except:
            pass
    dummy_col = []
    for i in categorical_var:
        if i in df.columns:
            dummy_col.append(i)
    new_df = pd.get_dummies(data=df,columns=dummy_col)
    #Setting up X,y
    lst=[]
    for i in new_df.columns:
        if i != target_var[0]:
            lst.append(i)

    X = new_df.loc[:,lst]
    y = new_df.loc[:,target_var[0]]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=float(size))
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    name=''
    flag = ''
    #Algorithm
    if algo=='logr':
        model = LogisticRegression().fit(X_train,y_train)
        y_pred = model.predict(X_test)
        name = 'logisticRegression'
        flag='c'
    elif algo=='dtc':
        model = DecisionTreeClassifier().fit(X_train,y_train)
        y_pred = model.predict(X_test)
        name = 'decisionTreeClassifier'
        flag='c'
    elif algo=='rfc':
        model = RandomForestClassifier().fit(X_train,y_train)
        y_pred = model.predict(X_test)
        name = 'randomForestClassifier'
        flag='c'
    elif algo=='lr':
        model = LinearRegression().fit(X_train,y_train)
        y_pred_ = model.predict(X_test)
        name = 'linearRegression'
        flag='r'
    elif algo=='pr':
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        model = LinearRegression().fit(X_poly,y_train)
        y_pred_ = model.predict(X_test)
        name='polynomialRegression'
        flag='r'
    elif algo=='dtr':
        model = DecisionTreeRegressor().fit(X_train,y_train)
        y_pred_ = model.predict(X_test)
        name = 'decisionTreeRegressor'
        flag='r'
    elif algo=='rfr':
        model = RandomForestRegressor().fit(X_train,y_train)
        y_pred_ = model.predict(X_test)
        name = 'randomForestRegressor'
        flag='r'
    #Saving model
    uniq_id = datetime.datetime.now().strftime("%c")
    uniq_id = ('').join(uniq_id.split())
    uniq_id = re.sub(":","",uniq_id)
    filename = "{}_{}.pkl".format(name,uniq_id)
    model_path = "static/{}_{}.pkl".format(name,uniq_id) 
    joblib.dump(model, model_path)
    #Metrics
    if flag=='c':
        cm = confusion_matrix(y_test,y_pred)
        df_cm = pd.DataFrame(cm,index=[i for i in set(new_df[target_var[0]])],
                    columns=[i for i in set(new_df[target_var[0]])])
        precision,recall,fscore,support=score(y_test,y_pred,average='macro')
        a4_dims = (11.7, 8.27)
        fig, ax = plt.subplots(figsize=a4_dims)
        fig_ = sns.heatmap(df_cm,annot=True,ax=ax)
        fig_copy = fig_.get_figure()
        uniq_id = datetime.datetime.now().strftime("%c")
        uniq_id = ('').join(uniq_id.split())
        uniq_id = re.sub(":","",uniq_id)
        fig_path = "cm_{}.png".format(uniq_id) 
        fig_copy.savefig("static/{}".format(fig_path))
        return [fig_path,round(precision,2),round(recall,2),round(fscore,2)],flag,model_path
    elif flag=='r':
        r2 = r2_score(y_test,y_pred_)
        mae = mean_absolute_error(y_test,y_pred_)
        mse = mean_squared_error(y_test,y_pred_)
        return [round(r2,2),round(mae,2),round(mse,2)],flag,model_path
        
        






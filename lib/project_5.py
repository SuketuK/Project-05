from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Lasso

def connect_to_postgres():
    url = 'postgresql://dsi:correct horse battery staple@joshuacook.me:5432'
    #engine = create_engine("postgresql://{}:{}@{}:{}/{}".format('dsi', 'correct horse battery staple', url, port, database))
    engine = create_engine(url)
    return engine
    
def load_data_from_database():
    connection  = connect_to_postgres()
    sql_query = """
    select * from madelon
    """
    madelon_df = pd.read_sql(sql_query,con=connection)
    
    return madelon_df

def make_data_dict(X,y,test_size_per=0.5,random_state=None):
    """ This function takes a pandas DataFrame and splits the data in to train and test data. The random state if not give will be ignored. The test data size is assumed to be 50%. 
    It creates a data dictionary and returns X_train, X_test, y_train and y_test."""
    #y = df["label"]
    #X = df.drop("label",axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=test_size_per,random_state=random_state)
    return {'X_train' : X_train,
            'y_train' : y_train,
            'X_test'  : X_test,
            'y_test'  : y_test}

def general_transformer(transformer,data_dict):
    """ This function takes any transformer e.g. StandardScaler, performs fit i.e. compute the mean and std to be used for later scaling. and then tranforms the train and test data i.e. perform standardization by centering and scaling. 
    It returns an updated data dictionary with transformer """
    if 'processes' in data_dict.keys():
        data_dict['processes'].append(transformer)
    else:
        data_dict['processes'] = [transformer]
        
        # Perform standadization by centering and scaling (i.e. transform). Scaling the data improves the accuracy and is also 
        # necessary for proper regularization and parameter tuning.
    transformer.fit(data_dict['X_train'],data_dict['y_train'])
    
    data_dict["X_train"] = transformer.transform(data_dict["X_train"])
    data_dict["X_test"] = transformer.transform(data_dict["X_test"])
    
    return data_dict

def general_model(model, data_dict):
    """ This function takes any model e.g. LogisticRegression, performs fit, get the accuracy score, updates the dictionary with model and score. Return the dictionary."""
    # Fit the model
    if 'processes' in data_dict.keys():
        data_dict['processes'].append(model)
    else:
        data_dict['processes'] = [model]
    
    model.fit(data_dict["X_train"], data_dict["y_train"])
    
    data_dict["train_score"] = model.score(data_dict["X_train"], data_dict["y_train"])
    data_dict["test_score"] = model.score(data_dict["X_test"], data_dict["y_test"])
    
    return data_dict
def Get_Data_from_Step1():
    """ This functions returns data dictionary with split data"""
    madelon_df = load_data_from_database()
    del madelon_df["index"]
    # Define X and y
    y = madelon_df["label"]
    X = madelon_df.drop("label",axis=1)
    data_dict= make_data_dict(X,y,0.5,random_state=42)



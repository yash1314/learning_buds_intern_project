import os
import sys
import joblib

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler

def save_obj(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        logging.info('Error occured in utils save_obj')
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    minmax = MinMaxScaler()

    X_train = pd.DataFrame(minmax.fit_transform(X_train), columns=minmax.get_feature_names_out())
    X_test = pd.DataFrame(minmax.transform(X_test), columns = minmax.get_feature_names_out())

    try:

        model = models

        # Train model
        model.fit(X_train,y_train)

        # Predict Testing data
        y_test_pred = model.predict(X_test)

        # Get R2 scores for train and test data
        test_model_score = recall_score(y_test,y_test_pred)

        return model, test_model_score

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return joblib.load(file_obj)
        
    except Exception as e:
        logging.info('Exception occured in load_obj in utils')
        raise CustomException(e, sys)
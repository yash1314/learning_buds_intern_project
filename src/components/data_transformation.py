import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesRegressor


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from ..exception import CustomException
from ..logger import logging
from ..utils import * 

from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    transformed_data_file_path = os.path.join('artifact', 'transformed_data.csv')
    

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

   
    def initiate_data_transformation(self, data_path):
        try : 
            ## reading the data
            df = pd.read_csv(data_path)

            # lower casing the column names
            df.columns = df.columns.str.lower() # renaming columns

            ## Capping 0's values of various columns
             
            lower_bound = df.glucose.mean() - 2.5 * df.glucose.std()
            df.glucose[df.glucose < lower_bound] = lower_bound
            
            lower_bound_preg = df.bloodpressure.mean() - 2 * df.bloodpressure.std()
            df.bloodpressure[df.bloodpressure == 0] = lower_bound_preg

            lower_bound_bmi = df.bmi.mean() - 1.5 * df.bmi.std()
            df.bmi[df.bmi == 0] = lower_bound_bmi

            logging.info('df data transformation completed')
            logging.info(f' transformed df data head: \n{df.head().to_string()}')

            df.to_csv(self.data_transformation_config.transformed_data_file_path, index = False, header= True)
            logging.info("transformed data is stored")

           
            ## splitting the data into training and target data
            X = df.drop(['outcome', 'skinthickness'], axis = 1)
            y = df['outcome']

             ## accessing the feature importance.
            select = ExtraTreesRegressor()
            select.fit(X, y)

            plt.figure(figsize=(10, 6))
            fig_importances = pd.Series(select.feature_importances_, index=X.columns)
            fig_importances.nlargest(20).plot(kind='barh')
        
            ## specify the path to the "visuals" folder using os.path.join
            visuals_folder = 'visuals'
            if not os.path.exists(visuals_folder):
                os.makedirs(visuals_folder)

            ## save the plot in the visuals folder
            plt.savefig(os.path.join(visuals_folder, 'feature_importance_plot.png'))
            logging.info('Saving feature importance figure is successful')

            ## further Splitting the data.
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 32, test_size=0.21, shuffle = True) 
            logging.info('final splitting the data is successful')
            
            ## returning splitted data and data_path.
            return (
                X_train, 
                X_test, 
                y_train, 
                y_test,
                self.data_transformation_config.transformed_data_file_path
            )
        
        except Exception as e:
            logging.info('error occured in the initiate_data_transformation')
            raise CustomException(e, sys)
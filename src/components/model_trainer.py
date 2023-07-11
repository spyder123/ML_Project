# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from sklearn import metrics
from src.utils import save_object
from src.utils import evaluate_model
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LogisticRegression':LogisticRegression(),
            
            'XGBClassifier': XGBClassifier(),
            'XGBClassifier_Best_Param': XGBClassifier(learning_rate=0.01,max_depth=5,n_estimators=200),
            'RandomForestClassifier':RandomForestClassifier(),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'DecisionTreeClassifier_Best_Param':DecisionTreeClassifier(criterion = 'gini', max_depth =4, splitter ='best')
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            # 'LogisticRegression_Best_Param':LogisticRegression(C=100),
            # 'RandomForestRegressor_Best_Param':RandomForestRegressor(n_estimators=100,max_depth=10,max_features='log2'),
            
            # 'XGBClassifier_Best_Param': XGBClassifier(learning_rate=0.01,max_depth=5,n_estimators=200),
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
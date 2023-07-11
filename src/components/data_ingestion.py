import os
import sys
from src.logger import logging
import numpy as np
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import statsmodels.api as sm
from src.components.data_transformation import DataTransformation
from patsy import dmatrices

## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv(os.path.join('notebooks/datafiles','hypothyroid.csv'))
            df["binaryClass"]=df["binaryClass"].map({"P":0,"N":1})
            df=df.replace({"t":1,"f":0})
            df['TSH']=df['TSH'].replace({"?":0})
            df['T3']=df['T3'].replace({"?":0})
            df['TT4']=df['TT4'].replace({"?":0})
            df['T4U']=df['T4U'].replace({"?":0})
            df['FTI']=df['FTI'].replace({"?":0})
            df=df.replace({"?":np.NAN})
            df=df.replace({"F":1,"M":0})
            cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            
            df['sex'].fillna(df['sex'].mode()[0], inplace=True)            

            logging.info('Dataset read as pandas Dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
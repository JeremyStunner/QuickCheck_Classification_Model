import sys
import os
import pandas as pd
from src.exception import CustomException
from src.Utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path= 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)




class CustomData:
    def __init__(  self,
        Account_type: int,
        Duration_of_Credit_month: int,
        Payment_Status_of_Previous_Credit: int,
        Purpose: int,
        Credit_Amount: int,
        Savings_type: int,
        Length_of_current_employment: int,
        Instalment_percent: int,
        Marital_Status: int,
        Guarantors: int,
        Duration_in_Current_address: int,
        Most_valuable_available_asset: int,
        Age: int,
        Concurrent_Credits: int,
        Type_of_apartment: int,
        No_of_Credits_at_this_Bank: int,
        Occupation: int,
        No_of_dependents: int,
        Telephone: int,
        Foreign_Worker: int):

        self.Account_type = Account_type
        self.Duration_of_Credit_month = Duration_of_Credit_month
        self.Payment_Status_of_Previous_Credit = Payment_Status_of_Previous_Credit
        self.Purpose = Purpose
        self.Credit_Amount = Credit_Amount
        self.Savings_type = Savings_type
        self.Length_of_current_employment = Length_of_current_employment
        self.Instalment_percent = Instalment_percent
        self.Marital_Status = Marital_Status
        self.Guarantors = Guarantors
        self.Duration_in_Current_address = Duration_in_Current_address
        self.Most_valuable_available_asset = Most_valuable_available_asset
        self.Age = Age
        self.Concurrent_Credits = Concurrent_Credits
        self.Type_of_apartment = Type_of_apartment
        self.No_of_Credits_at_this_Bank = No_of_Credits_at_this_Bank
        self.Occupation = Occupation
        self.No_of_dependents = No_of_dependents
        self.Telephone = Telephone
        self.Foreign_Worker = Foreign_Worker

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Account_type": [self.Account_type],
                "Duration_of_Credit_month": [self.Duration_of_Credit_month],
                "Payment_Status_of_Previous_Credit": [self.Payment_Status_of_Previous_Credit],
                "Purpose": [self.Purpose],
                "Credit_Amount": [self.Credit_Amount],
                "Savings_type": [self.Savings_type],
                "Length_of_current_employment": [self.Length_of_current_employment],
                "Instalment_percent": [self.Instalment_percent],
                "Marital_Status": [self.Marital_Status],
                "Guarantors": [self.Guarantors],
                "Duration_in_Current_address": [self.Duration_in_Current_address],
                "Most_valuable_available_asset": [self.Most_valuable_available_asset],
                "Age": [self.Age],
                "Concurrent_Credits": [self.Concurrent_Credits],
                "Type_of_apartment": [self.Type_of_apartment],
                "No_of_Credits_at_this_Bank": [self.No_of_Credits_at_this_Bank],
                "Occupation": [self.Occupation],
                "No_of_dependents": [self.No_of_dependents],
                "Telephone": [self.Telephone],
                "Foreign_Worker": [self.Foreign_Worker]
                            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.Utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "logistic Regression": LogisticRegression(),
                "KNN": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier()
            }     

            model_report:dict=evaluate_models(
                x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models
            )  

            # To get the best model accuracy from the dict
            best_model_accuracy = max(sorted(model_report.values()))

            # To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_accuracy)
            ]

            best_model = models[best_model_name]

            if best_model_accuracy<0.6:
                raise CustomException("No good model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
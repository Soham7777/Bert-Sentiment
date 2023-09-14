import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,f1_score,accuracy_score 
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    
    try:
        data = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None)
        data.columns = ["Text", "Sentiment"]

    except Exception as e:
        logger.exception(
            "Unable to retreive data. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train,X_test,y_train,y_test = train_test_split(data['Text'],data['Sentiment'],test_size = 0.25)


    kernel_val = str(sys.argv[1]) # can experiment with ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

    with mlflow.start_run():

        # Create a pipeline with CountVectorizer and SVM
        text_clf = Pipeline([
         ('vectorizer', CountVectorizer()),
         ('classifier', SVC(kernel=kernel_val)) ])
        
        # Fit the pipeline on your data
        text_clf.fit(X_train, y_train)

        predicted_qualities = text_clf.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        # Calculate F1-score
        f1 = f1_score(y_test, predicted_qualities)

        print(f"SVM model kernel = {kernel_val}")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")
        print(f"  F1: {f1}  ")

        mlflow.log_param("kernel used",kernel_val)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("F1", f1 )

        predictions = text_clf.predict(X_train)
        signature = infer_signature(X_train, predictions)

        #connect remotely 
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(
                text_clf, "model", registered_model_name="", signature=signature
            )

        else:
            mlflow.sklearn.log_model(text_clf, "model", signature=signature)

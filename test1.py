import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
import mlflow
import mlflow.sklearn
import logging
import sys
import warnings
from urllib.parse import urlparse

np.random.seed(0)
def eval_metrics(actual, pred):
    precision = precision_score(y_test, y_pred,average="weighted")
    recall = recall_score(y_test, y_pred,average="weighted")
    f1 = f1_score(y_test, y_pred, average="micro")
    accuracy = accuracy_score(y_test, y_pred)
    return precision,recall,f1,accuracy
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"fetal_health.csv"))
#data.head()

X=data.drop(["fetal_health"],axis=1)
y=data["fetal_health"]

#Set up a standard scaler for the features
col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X_df= s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names) 
X_train, X_test, y_train,y_test = train_test_split(X_df,y,test_size=0.3,random_state=42)
#X_train, X_test, y_train,y_test = train_test_split(X_df,y,test_size=0.3,random_state=42)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    mlflow.set_tracking_uri("http://localhost:8000")
    classifiers=[LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(), SVC()]
    ne = int(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    md = int(sys.argv[2])  if len(sys.argv) > 2 else 0.5
    with mlflow.start_run(run_name="Fetal Health"):
        classifier=RandomForestClassifier(n_estimators=ne,max_depth=md,random_state=42)
        model_lr=classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        (precision,recall,f1,accuracy)=eval_metrics(y_test, y_pred)

        print("  precision: %s" % precision)
        print("  recall: %s" % recall)
        print("  f1: %s" % f1)
        print(" accuracy:%s"% accuracy)

        mlflow.log_param("n_estimators", ne)
        mlflow.log_param("max_depth", md)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model_lr, "model", registered_model_name="Random_forest")
        else:
            mlflow.sklearn.log_model(model_lr, "model")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')

def Preprocess(df, xNames, yName, test_size=0.2, random_state=4):
    X = np.asarray(df[xNames])
    y = np.asarray(df[yName].astype('int'))
    Xnorm = StandardScaler().fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        Xnorm, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def ModelEval(Model, X_test, y_test):
    yhat_prob = Model.predict_proba(X_test)
    return log_loss(y_test, yhat_prob)



def  main():
    # Load dataset
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
    churn_df = pd.read_csv(url)
    ogInputFeatures = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
    outputFeature = 'churn'
    features = [ogInputFeatures+['callcard'],
                ogInputFeatures+['wireless'],
                ogInputFeatures+['callcard']+['wireless'],
                [x for x in ogInputFeatures if x!='equip'],
                [x  for x in ogInputFeatures if x!='income' and x!='employ']
                ]
    featureDetails = ['callcard included',
                      'wireless included',
                      'callcard and wireless included',
                      'equip excluded',
                      'income and employ excluded'
                      ]
    for feature, detail in zip(features, featureDetails):
        X_train, X_test, y_train, y_test = Preprocess(churn_df, feature, outputFeature)
        LR = LogisticRegression().fit(X_train, y_train)
        logLoss = ModelEval(LR, X_test, y_test)
        print(f"Log loss error with {detail}: {logLoss}")

if __name__ == '__main__':
    # print("All module imported sucessfully.")
    # print(churn_df.head(5))
    main()


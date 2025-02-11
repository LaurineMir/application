"""
Prediction de la survie d'un individu sur le Titanic
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pathlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import time
import os

os.chdir('/home/coder/work/ensae-reproductibilite-application')
TrainingData = pd.read_csv('data.csv')

TrainingData.head()


TrainingData['Ticket'].str.split("/").str.len()

TrainingData['Name'].str.split(",").str.len()

n_trees = 20
max_depth =None
max_features='sqrt'

TrainingData.isnull().sum()


## Un peu d'exploration et de feature engineering

### Statut socioéconomique

fig, axes=plt.subplots(1,2, figsize=(12, 6)) #layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass=sns.countplot(data=TrainingData, x ="Pclass",    ax=axes[0]).set_title("fréquence des Pclass")
fig2_pclass=sns.barplot(data=TrainingData, x= "Pclass",y= "Survived", ax=axes[1]).set_title("survie des Pclass")


### Age

sns.histplot(data= TrainingData, x='Age',bins=15, kde=False    )    .set_title("Distribution de l'âge")
plt.show()

## Encoder les données imputées ou transformées.
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

from dotenv import load_dotenv
import argparse
import os
load_dotenv()

parser = argparse.ArgumentParser(description="taille de l'arbre")
parser.add_argument(
    "--n_trees", type=int, default=20, help="taille de l'arbre"
)
args = parser.parse_args()
print(args.n_trees)

N_TREES = args.n_trees
MAX_DEPTH = None
MAX_FEATURES = "sqrt"
JETON_API = os.environ["JETON_API"]


if JETON_API.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")


def exploration_code(colonne,sign):
    colonne.str.split(sign).str.len()

def split_and_save_data(data, target_column, test_size, train_path="train.csv", test_path="test.csv"):
    y=data[target_column]
    X = data.drop(target_column, axis="columns")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    pd.concat([X_train, y_train]).to_csv(train_path)
    pd.concat([X_test, y_test]).to_csv(test_path)
    return X_train,X_test,y_train,y_test

# IMPORT ET EXPLORATION DONNEES --------------------------------

TrainingData = pd.read_csv("data.csv")

exploration_code(TrainingData["Ticket"],"/")
exploration_code(TrainingData["Name"],",")

#TrainingData["Ticket"].str.split("/").str.len()
#TrainingData["Name"].str.split(",").str.len()

TrainingData.isnull().sum()

# Statut socioéconomique
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")

# Age
sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()


# SPLIT TRAIN/TEST --------------------------------

# On _split_ notre _dataset_ d'apprentisage
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.

X_train,X_test,y_train,y_test=split_and_save_data(TrainingData,"Survived",0.1)

# PIPELINE ----------------------------
def pipeline_model(n_trees,numeric,categorical)
# Définition des variables
numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

# Variables numériques
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

# Variables catégorielles
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

# Pipeline
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=N_TREES)),
    ]
)


# ESTIMATION ET EVALUATION ----------------------

pipe.fit(X_train, y_train)

# score
rdmf_score = pipe.score(X_test, y_test)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")

print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))

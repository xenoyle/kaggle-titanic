import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


""" DATA INITIALIZATION """

df = pd.read_csv("train.csv")

x = df.drop(["PassengerId", "Survived", "Ticket", "Cabin"], axis=1)
y = df["Survived"]


""" FEATURE ENGINEERING """

# Blank Age Imputation using Titles

x["Title"] = x["Name"].str.extract('([A-Za-z]+)\.')

print(f"Titles: {x['Title'].unique()}")

x["Title"] = x["Title"].replace(['Mlle', 'Ms'], 'Miss')
x["Title"] = x["Title"].replace('Mme', 'Mrs')
x["Title"] = x["Title"].replace(['Lady', 
                                 'Countess', 
                                 'Capt', 
                                 'Col', 
                                 'Don', 
                                 'Dr', 
                                 'Major', 
                                 'Rev', 
                                 'Sir', 
                                 'Jonkheer', 
                                 'Dona'], 'Rare')

print(f"Titles: {x['Title'].unique()}")

x["Age"] = x["Age"].fillna(x.groupby("Title")["Age"].transform("mean"))

x = x.drop(["Name", "Title"], axis=1)

# Merging SibSp and Parch to represent family size

x["FamilySize"] = x["SibSp"] + x["Parch"]

x = x.drop(["SibSp", "Parch"], axis=1)

# Logarithmic transform to Fare, which is a bit exponential, skewing results

x["Fare"] = np.log1p(x["Fare"])


""" PRE-PROCESSING """

x_num = x.drop(["Sex", "Embarked"], axis=1)
x_cat = x[["Sex", "Embarked"]]

encoder = OneHotEncoder(sparse_output=False)
imp_num = SimpleImputer(strategy="mean")
imp_cat = SimpleImputer(strategy="most_frequent")
scaler = StandardScaler()

x_num = imp_num.fit_transform(x_num)
x_cat = imp_cat.fit_transform(x_cat)

x_num = scaler.fit_transform(x_num)

x_cat = encoder.fit_transform(x_cat)

x = np.append(x_num, x_cat, axis=1)


""" NEURAL NETWORK """

predictors = x
target = y
n_cols = predictors.shape[1]

model = Sequential()

model.add(Input(shape=(n_cols,)))

model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(1, activation="sigmoid"))

optimizer = Adam()

early_stop = EarlyStopping(monitor="val_loss", 
                           patience=10, 
                           restore_best_weights=True)

model.compile(optimizer=optimizer, 
              loss="binary_crossentropy", 
              metrics=["accuracy"])

model.fit(predictors, 
          target, 
          epochs=200, 
          validation_split=0.2, 
          callbacks=early_stop)


""" PREDICTIONS """

df_test = pd.read_csv("test.csv")

passengerid_col = df_test["PassengerId"]

x_test = df_test.drop(["PassengerId", "Ticket", "Cabin"], axis=1)

# Feature engineering once again

x_test["Title"] = x_test["Name"].str.extract('([A-Za-z]+)\.')

print(f"Titles: {x_test['Title'].unique()}")

x_test["Title"] = x_test["Title"].replace(['Mlle', 'Ms'], 'Miss')
x_test["Title"] = x_test["Title"].replace('Mme', 'Mrs')
x_test["Title"] = x_test["Title"].replace(['Lady', 
                                           'Countess', 
                                           'Capt', 
                                           'Col', 
                                           'Don', 
                                           'Dr', 
                                           'Major', 
                                           'Rev', 
                                           'Sir', 
                                           'Jonkheer', 
                                           'Dona'], 'Rare')

print(f"Titles: {x_test['Title'].unique()}")

x_test["Age"] = x_test["Age"].fillna(x_test.groupby("Title")["Age"].transform("mean"))

x_test = x_test.drop(["Name", "Title"], axis=1)

# Family Size

x_test["FamilySize"] = x_test["SibSp"] + x_test["Parch"]

x_test = x_test.drop(["SibSp", "Parch"], axis=1)

# Logarithmic transform to Fare

x_test["Fare"] = np.log1p(x_test["Fare"])

x_test_num = x_test.drop(["Sex", "Embarked"], axis=1)
x_test_cat = x_test[["Sex", "Embarked"]]

x_test_num = imp_num.transform(x_test_num)
x_test_cat = imp_cat.transform(x_test_cat)

x_test_num = scaler.transform(x_test_num)

x_test_cat = encoder.transform(x_test_cat)

x_test = np.append(x_test_num, x_test_cat, axis=1)

predictions = model.predict(x_test, verbose=2)

bool_result = predictions > 0.5
result = bool_result.astype(int).flatten()

submission = pd.DataFrame(data={"PassengerId":passengerid_col, "Survived":result})

submission.to_csv("submission.csv", index=False)





import pandas as pd

test = pd.read_csv("test.csv")
test_shape = test.shape

train = pd.read_csv("train.csv")
train_shape = train.shape

import matplotlib.pyplot as plt

sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()

pclass_pivot = train.pivot_table(index = "Pclass", values = "Survived")
pclass_pivot.plot.bar()
plt.show()

#we know from basic experience that in general sex and age were deciding factors
#amongst titanic survival, as well as the wealth of the patron. the age
#column will need some work because of its sparseness as well as how the data is
#presented

train["Age"].describe()
survived = train[train["Survived"] ==  1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=.5, color = 'red', bins= 50)
died["Age"].plot.hist(alpha = .5, color = 'blue', bins = 50)
plt.legend(['Survived','Died'])
plt.show()

def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
labels = ['Missing','Infant','Child','Teenager','Young Adult','Adult','Senior']
train = process_age(train, cut_points, labels)
test = process_age(test, cut_points, labels)

age_pivot = train.pivot_table(index = 'Age_categories', values = 'Survived')
age_pivot.plot.bar()

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train,"Pclass")
test = create_dummies(test,"Pclass")

train = create_dummies(train, "Sex")
test = create_dummies(test, "Sex")

train = create_dummies(train, "Age_categories")
test = create_dummies(test, "Age_categories")

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
columns = ['Pclass_1', 'Pclass_2','Pclass_3','Sex_female','Sex_male', 'Age_categories_Missing', 'Age_categories_Infant','Age_categories_Child'
          ,'Age_categories_Teenager','Age_categories_Young Adult','Age_categories_Adult','Age_categories_Senior']

lr.fit(train[columns], train['Survived'])

#we now have our first attempt at a model but since we used the whole training set on it we can't
#evaluate it unless we evaluate an overfit model
#we are going to split our training data to do some evaluation

from sklearn.model_selection import train_test_split
all_X = train[columns]
all_Y = train['Survived']

train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y, test_size = .2, random_state = 0)
lr.fit(train_X, train_Y)
from sklearn.metrics import accuracy_score
predictions = lr.predict(test_X)
accuracy = accuracy_score(test_Y, predictions)

lr = LogisticRegression()
lr.fit(train[columns], train["Survived"])
final_predictions = lr.predict(test[columns])

final_ids = test['PassengerId']
final_df = {
    'PassengerId': final_ids,
    'Survived': final_predictions
}

submission = pd.DataFrame(final_df)
submission.to_csv("submission.csv", index = False)


from sklearn.preprocessing import minmax_scale
# The holdout set has a missing value in the Fare column which
# we'll fill with the mean.
test["Fare"] = test["Fare"].fillna(train["Fare"].mean())

train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

train = create_dummies(train, "Embarked")
test = create_dummies(test, "Embarked")

columns_to_scale = ["SibSp","Parch","Fare"]
for column in columns_to_scale:
    train[column + "_scaled"] = minmax_scale(train[column])
    test[column+"_scaled"] = minmax_scale(test[column])

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_scaled']
lr = LogisticRegression()
lr.fit(train[columns],train['Survived'])

coefficients = lr.coef_

feature_importance = pd.Series(coefficients[0],
                               index=train[columns].columns)
feature_importance.plot.barh()
plt.show()

from sklearn.model_selection import cross_val_score

columns = ['Age_categories_Infant', 'SibSp_scaled', 'Sex_female', 'Sex_male',
       'Pclass_1', 'Pclass_3', 'Age_categories_Senior', 'Parch_scaled']

lr = LogisticRegression()
scores = cross_val_score(lr, train[columns], train["Survived"], cv = 10)
accuracy = scores.mean()
print(accuracy)


lr = LogisticRegression()
lr.fit(train[columns], train["Survived"])
predictions = lr.predict(test[columns])
submission_dict = {
    "PassengerId" : test["PassengerId"],
    "Survived": predictions
}
submission_frame = pd.DataFrame(submission_dict)
submission_frame.to_csv("submission_1.csv", index = False)

def process_fare(df, cut_points, label_names):
    df["Fare_categories"] = pd.cut(df["Fare"], cut_points, labels = label_names)
    return df
#this one did slightly better but we can start to incorporate fare and titles from the names and may get
#even better

cut_points = [0,12,50,100,1000]
label_names = ["0-12","12-50","50-100","100+"]
train = process_fare(train, cut_points, label_names)
test = process_fare(test, cut_points, label_names)

train = create_dummies(train, "Fare_categories")
test = create_dummies(test, "Fare_categories")

titles = {
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"
}

extracted_titles = train["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
train["Title"] = extracted_titles.map(titles)
extracted_titles_test = test["Name"].str.extract(' ([A-Za-z]+)\.',expand = False)
test["Title"] = extracted_titles_test.map(titles)

train["Cabin_type"] = train["Cabin"].str[0]
train["Cabin_type"] = train["Cabin_type"].fillna("Unknown")
test["Cabin_type"] = test["Cabin"].str[0]
test["Cabin_type"] = test["Cabin_type"].fillna("Unknown")

train = create_dummies(train, "Title")
test = create_dummies(test, "Title")
train = create_dummies(train, "Cabin_type")
test = create_dummies(test, "Cabin_type")

import numpy as np
import seaborn as sns

def plot_correlation_heatmap(df):
    corr = df.corr()

    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)


    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_categories_0-12',
       'Fare_categories_12-50','Fare_categories_50-100', 'Fare_categories_100+',
       'Title_Master', 'Title_Miss', 'Title_Mr','Title_Mrs', 'Title_Officer',
       'Title_Royalty', 'Cabin_type_A','Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D',
       'Cabin_type_E','Cabin_type_F', 'Cabin_type_G', 'Cabin_type_T', 'Cabin_type_Unknown']

plot_correlation_heatmap(train[columns])

from sklearn.feature_selection import RFECV
columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Young Adult',
       'Age_categories_Adult', 'Age_categories_Senior', 'Pclass_1', 'Pclass_3',
       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp_scaled',
       'Parch_scaled', 'Fare_categories_0-12', 'Fare_categories_50-100',
       'Fare_categories_100+', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
       'Title_Officer', 'Title_Royalty', 'Cabin_type_B', 'Cabin_type_C',
       'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G',
       'Cabin_type_T', 'Cabin_type_Unknown']

all_X = train[columns]
all_y = train["Survived"]

lr = LogisticRegression()
selector = RFECV(lr,cv = 10)
selector.fit(all_X, all_y)

optimized_columns = all_X.columns[selector.support_]

all_X = train[optimized_columns]
all_y = train["Survived"]

lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv= 10)
accuracy = scores.mean()

lr = LogisticRegression()
lr.fit(all_X, all_y)
holdout_predictions = lr.predict(test[optimized_columns])
submission_dict = {
    'PassengerId': test['PassengerId'],
    'Survived':holdout_predictions
}

submission = pd.DataFrame(submission_dict)
submission.to_csv("submission_2.csv", index = False)

#now we want to do more work with the model selection

train = train.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age_categories','Title','Fare_categories', 'Cabin_type'], axis = 1)
test = test.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age_categories','Title','Fare_categories', 'Cabin_type'], axis = 1)

all_X = train.drop(['Survived','PassengerId'], axis = 1)
all_y = train['Survived']
lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv = 10)
accuracy_lr = scores.mean()

import matplotlib.pyplot as plt
%matplotlib inline

def plot_dict(dictionary):
    pd.Series(dictionary).plot.bar(figsize=(9,6),
                                   ylim=(0.78,0.83),rot=0)
    plt.show()

knn_scores = dict()

for value in range(1,50,2):
    knn = KNeighborsClassifier(n_neighbors = value)
    scores = cross_val_score(knn, all_X, all_y, cv = 10)
    knn_scores[value] = scores.mean()

plot_dict(knn_scores)

knn = KNeighborsClassifier()
hyperparameters = {
    "n_neighbors": range(1,20,2),
    "weights": ["distance", "uniform"],
    "algorithm": ['brute'],
    "p": [1,2]
}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(knn, param_grid = hyperparameters, cv = 10)
grid.fit(all_X, all_y)

print(grid.best_params_)
print(grid.best_score_)
best_params = grid.best_params_
best_score = grid.best_score_

# Get missing columns in the training test
missing_cols = set( train.columns ) - set( test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test = test[train.columns]

test_no_ids = test.drop(['PassengerId','Survived'], axis = 1)
best_knn = grid.best_estimator_
test_predictions = best_knn.predict(test_no_ids)
prediction_dict = {
    'PassengerId' : test['PassengerId'],
    'Survived': test_predictions
}

submission_df = pd.DataFrame(prediction_dict)
submission_df.to_csv("submission_knn.csv", index = False)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state = 1)
scores = cross_val_score(clf, all_X, all_y, cv = 10)
accuracy_rf = scores.mean()

hyperparameters = {"criterion": ["entropy", "gini"],
                   "max_depth": [5, 10],
                   "max_features": ["log2", "sqrt"],
                   "min_samples_leaf": [1, 5],
                   "min_samples_split": [3, 5],
                   "n_estimators": [6, 9]
}

clf = RandomForestClassifier(random_state=1)
grid = GridSearchCV(clf,param_grid=hyperparameters,cv=10)

grid.fit(all_X, all_y)

best_params = grid.best_params_
best_score = grid.best_score_

best_rf = grid.best_estimator_
test_predictions = best_rf.predict(test_no_ids)
prediction_dict = {
    'PassengerId' : test['PassengerId'],
    'Survived': test_predictions
}

submission_df = pd.DataFrame(prediction_dict)
submission_df.to_csv("submission_rf.csv", index = False)

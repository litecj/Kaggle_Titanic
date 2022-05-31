from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import feature_engineering
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from pandas import Series

import pandas as pd

train = pd.read_csv(r'./data/train.csv')
test = pd.read_csv(r'./data/test.csv')

train = feature_engineering(train)
test = feature_engineering(test)

#  Pclass, Sex, Embarked, Title, AgeCategory, CabinCategory, FareCategory, Family, IsAlone, TicketCategory
data = train.drop('Survived', axis=1).values
# Survived
target = train['Survived'].values

'''
data는 특성(feature)이 되고, target은 정답(Label)이 됩니다. data는 Survived를 제외한 Pclass, Sex, Embarked, Title, AgeCategory, CabinCategory, FareCategory, Family, IsAlone, TicketCategory 정보를 가지는 값입니다. target은 Survived 값입니다.

이제 data를 이용하여 Survived를 예측한 결과를 target과 비교하여 정확도를 판단해 보겠습니다.
'''

# test_size: 분리 비율 설정.
# stratify: 분리 기준이 될 데이터
# random_state: 랜덤 seed
x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size=0.4, stratify=target, random_state=0)


def random_forest_1():
    rf = RandomForestClassifier(n_estimators=50, criterion="entropy", max_depth=5, oob_score=True, random_state=10)
    rf.fit(x_train, y_train)
    prediction = rf.predict(x_valid)

    length = y_valid.shape[0]
    accuracy = accuracy_score(prediction, y_valid)
    print(f'총 {length}명 중 {accuracy * 100:.3f}% 정확도로 생존을 맞춤')

    # 결과
    #총 357명 중 83.473% 정확도로 생존을 맞춤

def random_forest_2():
    RF_classifier = RandomForestClassifier()

    RF_paramgrid = {
        'max_depth': [6, 8, 10, 15],
        'n_estimators': [50, 100, 300, 500, 700, 800, 900],
        'max_features': ['sqrt'],
        'min_samples_split': [2, 7, 15, 30],
        'min_samples_leaf': [1, 15, 30, 60],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    k_fold = StratifiedKFold(n_splits=5)
    RF_classifiergrid = GridSearchCV(RF_classifier, param_grid=RF_paramgrid, cv=k_fold, scoring="accuracy", n_jobs=-1,
                                     verbose=1)

    RF_classifiergrid.fit(x_train, y_train)

    rf = RF_classifiergrid.best_estimator_

    '''
        Fitting 5 folds for each of 1792 candidates, totalling 8960 fits
        [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
        [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   11.1s
        [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   39.3s
        [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  1.4min
        [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:  2.6min
        [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed:  4.0min
        [Parallel(n_jobs=-1)]: Done 1784 tasks      | elapsed:  5.5min
        [Parallel(n_jobs=-1)]: Done 2434 tasks      | elapsed:  7.1min
        [Parallel(n_jobs=-1)]: Done 3184 tasks      | elapsed:  9.2min
        [Parallel(n_jobs=-1)]: Done 4034 tasks      | elapsed: 11.5min
        [Parallel(n_jobs=-1)]: Done 4984 tasks      | elapsed: 14.3min
        [Parallel(n_jobs=-1)]: Done 6034 tasks      | elapsed: 17.3min
        [Parallel(n_jobs=-1)]: Done 7184 tasks      | elapsed: 20.3min
        [Parallel(n_jobs=-1)]: Done 8434 tasks      | elapsed: 23.7min
        [Parallel(n_jobs=-1)]: Done 8960 out of 8960 | elapsed: 25.1min finished
    '''
    # Best Score
    print(RF_classifiergrid.best_score_)
    0.8352059925093633

    # Best Parameter
    print(RF_classifiergrid.best_params_)
    {'bootstrap': True, 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 7,
     'n_estimators': 1200}

    # Best Model
    print(RF_classifiergrid.best_estimator_)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=8, max_features='sqrt', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=7,
                           min_weight_fraction_leaf=0.0, n_estimators=1200,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)

def check():
    feature_importance = random_forest_2.rf.feature_importances_
    fi = Series(feature_importance, index=train.drop(['Survived'], axis=1).columns)

    plt.figure(figsize=(8, 8))
    fi.sort_values(ascending=True).plot.barh()
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.show()
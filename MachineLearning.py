from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import feature_engineering
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

rf = RandomForestClassifier(n_estimators=50, criterion="entropy", max_depth=5, oob_score=True, random_state=10)
rf.fit(x_train, y_train)
prediction = rf.predict(x_valid)

length = y_valid.shape[0]
accuracy = accuracy_score(prediction, y_valid)
print(f'총 {length}명 중 {accuracy * 100:.3f}% 정확도로 생존을 맞춤')

# 결과
#총 357명 중 83.473% 정확도로 생존을 맞춤
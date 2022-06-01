from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from MachineLearning import train

# data 분리
data = train.drop('Survived', axis=1).values
target = train['Survived'].values
x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size=0.4, stratify=target, random_state=0)

# 모델 적용 함수
def ml_fit(model):
    model.fit(x_train, y_train)
    prediction = model.predict(x_valid)
    accuracy = accuracy_score(prediction, y_valid)
    print(model)
    print(f'총 {y_valid.shape[0]}명 중 {accuracy * 100:.3f}% 정확도로 생존을 맞춤')
    return model

# 기본 설정으로만 테스트
model = ml_fit(RandomForestClassifier(n_estimators=100))
model = ml_fit(LogisticRegression(solver='lbfgs'))
model = ml_fit(SVC(gamma='scale'))
model = ml_fit(KNeighborsClassifier())
model = ml_fit(GaussianNB())
model = ml_fit(DecisionTreeClassifier())

# 총 357명 중 79.832% 정확도로 생존을 맞춤
model = ml_fit(RandomForestClassifier(n_estimators=100))

# 총 357명 중 78.711% 정확도로 생존을 맞춤
model = ml_fit(LogisticRegression(solver='lbfgs'))

# 총 357명 중 79.832% 정확도로 생존을 맞춤
model = ml_fit(SVC(gamma='scale'))

# 총 357명 중 75.910% 정확도로 생존을 맞춤
model = ml_fit(KNeighborsClassifier())

# 총 357명 중 73.669% 정확도로 생존을 맞춤
model = ml_fit(GaussianNB())

# 총 357명 중 77.031% 정확도로 생존을 맞춤
model = ml_fit(DecisionTreeClassifier())

# 총 357명 중 81.513% 정확도로 생존을 맞춤
model = ml_fit(RandomForestClassifier(n_estimators=50, criterion="entropy", max_depth=5, oob_score=True, random_state=10))


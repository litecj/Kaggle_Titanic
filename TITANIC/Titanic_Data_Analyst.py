import numpy as np
import pandas as pd
import icecream as ic
import matplotlib.pyplot as plt

test_df = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')

def data_check():
    # print(train_df.head())
    # print(train_df.info())

    # 결측치 제거, 분석 하는데 영향이 있을수 있기 때문에 제거 하거나 0으로 설정 한다.
    # print(train_df.isnull().sum())
    # age, cabin, embarked 컬럼에 결측치가 있는게 확인도었다.
    print(train['Sex'])


def show_pie_chart(df, col_name):
    colname_survived = survived_crosstab(df, col_name)
    pie_chart(colname_survived)
    return colname_survived


def survived_crosstab(df, col_name):
    '''col_name과 Survived간의 교차도표 생성'''
    feature_survived = pd.crosstab(df[col_name], df['Survived'])
    feature_survived.columns = feature_survived.columns.map({0:"Dead", 1:"Alive"})
    return feature_survived


def pie_chart(feature_survived):
    '''
    pie_chart 생성
    pcol, prow = 차트를 출력할 개수. pcol * prow 만큼의 차트 출력
    '''
    frows, fcols = feature_survived.shape
    pcol = 3
    prow = (frows/pcol + frows%pcol)
    plot_height = prow * 2.5
    plt.figure(figsize=(8, plot_height))

    for row in range(0, frows):
        plt.subplot(int(prow), int(pcol), int(row+1))

        index_name = feature_survived.index[row]
        plt.pie(feature_survived.loc[index_name], labels=feature_survived.loc[index_name].index, autopct='%1.1f%%')
        plt.title("{0}' survived".format(index_name))

    plt.show()


def preprocessing():
    '''
    이름은 그대로는 분석에 사용할 수 없지만 이름에서 호칭을 추출할 수 있습니다. 아래와 같은 이름에서 Mr., Miss. 같은 호칭을 추출할 수 있습니다.

        Braund, Mr. Owen Harris
        Heikkinen, Miss. Laina
     아래와 같이 정규식으로 탑승자의 호칭(Title)을 추출하겠습니다.
    '''
    train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.')
    train['Title'].value_counts()
    '''
    추출한 호칭에서 데이터가 많은 MR, Miss, Mrs, Master는 그대로 두고 나머지는 Other로 변경합니다. 동시에 프랑스어로 된 호칭도 변경합니다.
    '''
    train['Title'] = train['Title'].replace(
        ['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')
    train['Title'] = train['Title'].replace('Mlle', 'Miss')
    train['Title'] = train['Title'].replace('Mme', 'Mrs')
    train['Title'] = train['Title'].replace('Ms', 'Miss')
    train['Title'].value_counts()

    # 생존률을 비교해 보면 Miss, Mrs에 비해 Mr의 생존률이 낮은 것을 확인할 수 있습니다.


    '''
    나이와 생존률의 관계를 알아보겠습니다. 먼저 나이에 존재하는 결측치를 호칭과 나이로 구한 나이의 평균값으로 채워 줍니다. 
    그리고 나이를 qcut을 이용하여 8개의 구간으로 분리하여 나이 카테고리(AgeCategory)로 만들어 줍니다.
    '''
    meanAge = train[['Title', 'Age']].groupby(['Title']).mean()
    for index, row in meanAge.iterrows():
        nullIndex = train[(train.Title == index) & (train.Age.isnull())].index
        train.loc[nullIndex, 'Age'] = row[0]

    train['AgeCategory'] = pd.qcut(train.Age, 8, labels=range(1, 9))
    train.AgeCategory = train.AgeCategory.astype(int)

    # 나이가 어릴 수록 생존률이 약간 높은 것을 알 수 있습니다. 그리고 결측치가 많이 포함된 데이터는 생존률이 떨어지는 것을 알 수 있습니다.

    '''
    방 번호와 생존률의 관계를 알아보겠습니다. 방번호의 결측치는 N으로 채우고, 방 번호의 첫 영문자만 때어내서 선택한 뒤 숫자로 카테고리화 합니다.
    '''
    train.Cabin.fillna('N', inplace=True)
    train["CabinCategory"] = train["Cabin"].str.slice(start=0, stop=1)
    train["CabinCategory"] = train['CabinCategory'].map(
        {"N": 0, "C": 1, "B": 2, "D": 3, "E": 4, "A": 5, "F": 6, "G": 7, "T": 8})

    # 방 번호가 낮아 질 수록 생존률이 떨어지는 것을 알 수 있습니다.

    '''
    운임과 생존률의 관계를 알아 보겠습니다. 운임의 결측치는 0으로 채우고 qcut을 이용하여 8개의 구간으로 분리하여 카테고리로 만들어 줍니다.
    '''
    train.Fare.fillna(0)  # test.csv 데이터에 결측치가 존재함.
    train['FareCategory'] = pd.qcut(train.Fare, 8, labels=range(1, 9))
    train.FareCategory = train.FareCategory.astype(int)

    # 운임이 높을 수록 생존률이 높아지는 것을 알 수 있습니다.

    '''
    형제/자매 여부와 부모/자식 여부는 모두 더해서 가족의 숫자로 표현하고, 혼자인 사람 여부를 표현하는 변수(IsAlone)를 생성하겠습니다.
    '''
    train['Family'] = train['SibSp'] + train['Parch'] + 1
    train.loc[train["Family"] > 4, "Family"] = 5

    train['IsAlone'] = 1
    train.loc[train['Family'] > 1, 'IsAlone'] = 0

    # 4인 가족의 생존률이 가장 높고, 가족이 많으면 생존률이 낮아집니다. 혼자 승선한 사람의 생존률이 가족이 있는 사람보다 낮습니다.

    '''
    티켓 정보에 있는 영문자를 이용해서 생존률을 확인해 보겠습니다. 
    티켓정보를 공백을 기준으로 분할하고 마지막 숫자의 첫번째 글자만 분할합니다. factorize를 이용하여 카테고리화 합니다.
    '''
    # STON/O2. 3101282를 ['STON/O2.', '3101282']로 변경하고, '3101282'의 첫 번째 3을 선택
    train['TicketCategory'] = train.Ticket.str.split()  # 공백으로 분리
    train['TicketCategory'] = [i[-1][0] for i in train['TicketCategory']]  #
    train['TicketCategory'] = train['TicketCategory'].replace(['8', '9', 'L'], '8')
    train['TicketCategory'] = pd.factorize(train['TicketCategory'])[0] + 1

     # 숫자가 높을 수록 생존률이 높아 지는 것을 확인할 수 있습니다.

    return train


if __name__ == '__main__':
    preprocessing()
    sex = show_pie_chart(train,'Sex')
    embarked = show_pie_chart(train, 'Embarked')
    name = show_pie_chart(train, 'Title')
    age = show_pie_chart(train, 'AgeCategory')
    cabin= show_pie_chart(train, 'CabinCategory')
    fare = show_pie_chart(train, 'FareCategory')
    sibsp = show_pie_chart(train, 'Family')
    parch = show_pie_chart(train, 'IsAlone')
    ticket = show_pie_chart(train, 'TicketCategory')
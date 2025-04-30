import pandas as pd 
import numpy as np
import os
from app.domain.model.titanic_schema import TitanicSchema

class TitanicService:
    
    dataset = TitanicSchema()  

    def new_model(self, fname) -> pd.DataFrame: 
        # Docker 환경에서 작동하도록 상대 경로 사용
        # 현재 디렉토리 기준 상대 경로
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, 'stored_data', fname)
        
        # 파일 존재 여부 확인
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            
        print(f"데이터 파일 로드 중: {data_path}")
        return pd.read_csv(data_path)
    
    def preprocess(self, train_fname, test_fname) -> object: 
        print("------모델 전처리 시작------")
        # 데이터 로드
        train_df = self.new_model(train_fname)
        test_df = self.new_model(test_fname)
        
        # ID 보존
        self.dataset.id = test_df['PassengerId']
        
        # Survived 컬럼 분리
        y_train = train_df['Survived']
        train_df = train_df.drop('Survived', axis=1)
        
        # 불필요한 특성 제거
        drop_features = ['SibSp', 'Parch', 'Cabin', 'Ticket'] 
        train_df, test_df = self.drop_feature(train_df, test_df, *drop_features)
        
        # 이름에서 타이틀 추출 및 처리
        train_df, test_df = self.extract_title_from_name(train_df, test_df)
        title_mapping = self.remove_duplicate_title(train_df, test_df)
        train_df, test_df = self.title_nominal(train_df, test_df, title_mapping)
        train_df, test_df = self.drop_feature(train_df, test_df, 'Name')
        
        # 성별 변환 및 원본 컬럼 제거
        train_df, test_df = self.gender_nominal(train_df, test_df)
        train_df, test_df = self.drop_feature(train_df, test_df, 'Sex')
        
        # 승선 항구 처리
        train_df, test_df = self.embarked_nominal(train_df, test_df)
        
        # 나이 그룹화 및 원본 컬럼 제거
        train_df, test_df = self.age_ratio(train_df, test_df)
        train_df, test_df = self.drop_feature(train_df, test_df, 'Age')
        
        # 클래스 서수화
        train_df, test_df = self.pclass_ordinal(train_df, test_df)
        
        # 요금 서수화 및 원본 컬럼 제거
        train_df, test_df = self.fare_ordinal(train_df, test_df)
        train_df, test_df = self.drop_feature(train_df, test_df, "Fare")
        
        # 결과 저장
        self.dataset.train = train_df
        self.dataset.test = test_df
        self.dataset.label = y_train
        
        return self.dataset

    @staticmethod
    def drop_feature(train_df, test_df, *features) -> tuple: 
        for feature in features:
            train_df = train_df.drop(feature, axis=1)
            test_df = test_df.drop(feature, axis=1)
        return train_df, test_df
    
    @staticmethod 
    def remove_duplicate_title(train_df, test_df):
        titles = set(train_df['Title'].unique()) | set(test_df['Title'].unique())
        print(titles)
        title_mapping = {'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
        return title_mapping
    
    @staticmethod 
    def extract_title_from_name(train_df, test_df):
        train_df['Title'] = train_df['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
        test_df['Title'] = test_df['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
        return train_df, test_df
    
    @staticmethod
    def title_nominal(train_df, test_df, title_mapping):
        for df in [train_df, test_df]:
            df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            df['Title'] = df['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Mme'], 'Rare')
            df['Title'] = df['Title'].replace(['Mlle'], 'Mr')
            df['Title'] = df['Title'].replace(['Miss'], 'Ms')
            df['Title'] = df['Title'].fillna(0) 
            df['Title'] = df['Title'].map(title_mapping)
        return train_df, test_df
    
    @staticmethod 
    def pclass_ordinal(train_df, test_df):
        return train_df, test_df
     
    @staticmethod
    def gender_nominal(train_df, test_df):
        train_df['Gender'] = train_df['Sex'].map({'male': 0, 'female': 1})
        test_df['Gender'] = test_df['Sex'].map({'male': 0, 'female': 1})
        return train_df, test_df
    
    @staticmethod
    def age_ratio(train_df, test_df): 
        age_mapping = {'Unknown':0 , 'Baby': 1, 'Child': 2, 'Teenager' : 3, 'Student': 4,
                       'Young Adult': 5, 'Adult':6,  'Senior': 7}
        
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 
                  'Student', 'Young Adult', 'Adult', 'Senior']
        
        for df in [train_df, test_df]:
            df['Age'] = df['Age'].fillna(-0.5) 
            df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
            df['AgeGroup'] = df['AgeGroup'].map(age_mapping)
            print(df[['Age', 'AgeGroup']])

        return train_df, test_df
 
    @staticmethod
    def fare_ordinal(train_df, test_df): 
        for df in [train_df, test_df]:
            df['FareGroup'] = pd.qcut(df['Fare'], 4, labels={1,2,3,4})
        
        train_df = train_df.fillna({'FareGroup': 1})
        test_df = test_df.fillna({'FareGroup': 1})

        return train_df, test_df
    
    @staticmethod
    def embarked_nominal(train_df, test_df):
        train_df = train_df.fillna({'Embarked': 'S'})
        test_df = test_df.fillna({'Embarked': 'S'})
        train_df['Embarked'] = train_df['Embarked'].map({'S':1, 'C':2, 'Q':3})
        test_df['Embarked'] = test_df['Embarked'].map({'S':1, 'C':2, 'Q':3})

        return train_df, test_df



    

    



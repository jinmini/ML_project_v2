import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from app.domain.model.titanic_schema import TitanicSchema

class ModelingService:
    """
    머신러닝 모델 학습 및 평가를 담당하는 서비스 클래스
    """
    
    @staticmethod
    def create_k_fold(this):
        """K-Fold 교차 검증을 위한 KFold 객체 생성
        Args:
            this (TitanicSchema): 데이터셋
        Returns:
            tuple: 훈련 데이터, 레이블, KFold 객체
        """
        X_train = this.train  # 전처리된 훈련 데이터
        y_train = this.label  # 생존 레이블
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)  # 10개의 폴드로 분할, 셔플링
        return X_train, y_train, k_fold

    @staticmethod
    def create_random_variable(this):
        """모델 학습에 필요한 랜덤 변수 생성
        Args:
            this (TitanicSchema): 데이터셋
        Returns:
            dict: 모델별 성능 저장을 위한 딕셔너리
        """
        return {
            'logistic_regression': [],
            'decision_tree': [],
            'random_forest': [],
            'naive_bayes': [],
            'knn': [],
            'svm': []
        }
    
    @staticmethod
    def accuracy_by_logistic_regression(this):
        """로지스틱 회귀 모델(Baseline)을 사용한 예측 정확도 계산
        Args:
            this (TitanicSchema): 데이터셋
        Returns:
            float: 교차 검증 평균 정확도
        """
        X_train = this.train  # 전처리된 훈련 데이터
        y_train = this.label  # 생존 레이블
        
        # 파이프라인 생성: 표준화 + 로지스틱 회귀
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        # StratifiedKFold를 사용하여 교차 검증 (클래스 비율 유지)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 교차 검증 수행
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
        
        # 각 폴드별 정확도 출력
        print(f"각 폴드 정확도: {scores}")
        print(f"평균 정확도: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        
        return scores.mean()

    @staticmethod
    def accuracy_by_dtree(this):
        """결정트리 모델을 사용한 예측 정확도 계산
        Args:
            this (TitanicSchema): 데이터셋
        Returns:
            float: 교차 검증 평균 정확도
        """
        X_train, y_train, k_fold = ModelingService.create_k_fold(this)
        
        # 결정트리 모델 생성
        decision_tree = DecisionTreeClassifier(random_state=0)
        
        # 교차 검증 수행
        scores = cross_val_score(decision_tree, X_train, y_train, cv=k_fold, scoring='accuracy')
        
        # 각 폴드별 정확도 출력
        print(f"각 폴드 정확도: {scores}")
        print(f"평균 정확도: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        
        return scores.mean()

    @staticmethod
    def accuracy_by_random_forest(this):
        """랜덤 포레스트 모델을 사용한 예측 정확도 계산
        Args:
            this (TitanicSchema): 데이터셋
        Returns:
            float: 교차 검증 평균 정확도
        """
        X_train, y_train, k_fold = ModelingService.create_k_fold(this)
        
        # 랜덤 포레스트 모델 생성
        random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
        
        # 교차 검증 수행
        scores = cross_val_score(random_forest, X_train, y_train, cv=k_fold, scoring='accuracy')
        
        # 각 폴드별 정확도 출력
        print(f"각 폴드 정확도: {scores}")
        print(f"평균 정확도: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        
        return scores.mean()

    @staticmethod
    def accuracy_by_naive_bayes(this):
        """나이브 베이즈 모델을 사용한 예측 정확도 계산
        Args:
            this (TitanicSchema): 데이터셋
        Returns:
            float: 교차 검증 평균 정확도
        """
        X_train, y_train, k_fold = ModelingService.create_k_fold(this)
        
        # 나이브 베이즈 모델 생성
        naive_bayes = GaussianNB()
        
        # 교차 검증 수행
        scores = cross_val_score(naive_bayes, X_train, y_train, cv=k_fold, scoring='accuracy')
        
        # 각 폴드별 정확도 출력
        print(f"각 폴드 정확도: {scores}")
        print(f"평균 정확도: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        
        return scores.mean()

    @staticmethod
    def accuracy_by_knn(this):
        """K-최근접 이웃 모델을 사용한 예측 정확도 계산
        Args:
            this (TitanicSchema): 데이터셋
        Returns:
            float: 교차 검증 평균 정확도
        """
        X_train, y_train, k_fold = ModelingService.create_k_fold(this)
        
        # KNN 모델 생성 (neighbors=5가 일반적인 기본값)
        knn = KNeighborsClassifier(n_neighbors=5)
        
        # 교차 검증 수행
        scores = cross_val_score(knn, X_train, y_train, cv=k_fold, scoring='accuracy')
        
        # 각 폴드별 정확도 출력
        print(f"각 폴드 정확도: {scores}")
        print(f"평균 정확도: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        
        return scores.mean()

    @staticmethod
    def accuracy_by_svm(this):
        """서포트 벡터 머신을 사용한 예측 정확도 계산
        Args:
            this (TitanicSchema): 데이터셋
        Returns:
            float: 교차 검증 평균 정확도
        """
        X_train, y_train, k_fold = ModelingService.create_k_fold(this)
        
        # SVM 모델 생성
        svm = SVC(kernel='rbf', random_state=0)
        
        # 교차 검증 수행
        scores = cross_val_score(svm, X_train, y_train, cv=k_fold, scoring='accuracy')
        
        # 각 폴드별 정확도 출력
        print(f"각 폴드 정확도: {scores}")
        print(f"평균 정확도: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        
        return scores.mean()
    
    #  GradientBoostingClassifier로 구현된 모델
    @staticmethod
    def accuracy_by_gradient_boosting(this):
        """Gradient Boosting 모델을 사용한 예측 정확도 계산
        Args:
            this (TitanicSchema): 데이터셋
        Returns:
            float: 교차 검증 평균 정확도
        """
        X_train, y_train, k_fold = ModelingService.create_k_fold(this)
        
        # 그래디언트 부스팅 모델 생성 (최적화된 하이퍼파라미터 사용)
        gradient_boosting = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        
        # 교차 검증 수행
        scores = cross_val_score(gradient_boosting, X_train, y_train, cv=k_fold, scoring='accuracy')
        
        # 각 폴드별 정확도 출력
        print(f"각 폴드 정확도: {scores}")
        print(f"평균 정확도: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        
        return scores.mean()

    

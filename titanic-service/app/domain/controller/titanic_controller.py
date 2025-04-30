from app.domain.service.titanic_service import TitanicService
from app.domain.service.modeling_service import ModelingService
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier

class Controller :
     
    service = TitanicService()
    modeling_service = ModelingService()
    '''
    print(f'결정트리 활용한 검증 정확도 {None}')
    print(f'랜덤포레스트 활용한 검증 정확도 {None}')
    print(f'나이브베이즈 활용한 검증 정확도 {None}')
    print(f'KNN 활용한 검증 정확도 {None}')
    print(f'SVM 활용한 검증 정확도 {None}')
    '''

    def preprocess(self, train_fname, test_fname): #데이터 전처리
        return self.service.preprocess(train_fname, test_fname)
    
    def learning(self): #모델 학습
        """
        여러 머신러닝 모델을 학습하고 각 모델의 정확도를 계산
        
        Returns:
            dict: 모델별 정확도를 담은 사전
        """
        print("\n" + "="*50)
        print("📊 타이타닉 생존자 예측 모델 학습 시작 📊".center(50))
        print("="*50)
        this = self.service.dataset
        
        # 베이스라인 모델(로지스틱 회귀) 정확도 계산
        print("\n🔍 베이스라인 모델(로지스틱 회귀) 학습 중...")
        baseline_accuracy = self.modeling_service.accuracy_by_logistic_regression(this)
        print(f"✅ 베이스라인 모델 정확도: {baseline_accuracy:.4f}")
        
        # 각 알고리즘별 정확도 계산
        accuracy = {}
        accuracy['베이스라인(로지스틱 회귀)'] = baseline_accuracy
        
        print("\n🔍 의사결정 트리 모델 학습 중...")
        tree_acc = self.modeling_service.accuracy_by_dtree(this)
        accuracy['결정트리'] = tree_acc
        if tree_acc > baseline_accuracy:
            print(f"✅ 의사결정 트리 정확도: {tree_acc:.4f} (베이스라인 대비: {tree_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ 의사결정 트리 정확도: {tree_acc:.4f} (베이스라인 대비: {tree_acc - baseline_accuracy:+.4f})")
        
        print("\n🔍 랜덤 포레스트 모델 학습 중...")
        rf_acc = self.modeling_service.accuracy_by_random_forest(this)
        accuracy['랜덤포레스트'] = rf_acc
        if rf_acc > baseline_accuracy:
            print(f"✅ 랜덤 포레스트 정확도: {rf_acc:.4f} (베이스라인 대비: {rf_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ 랜덤 포레스트 정확도: {rf_acc:.4f} (베이스라인 대비: {rf_acc - baseline_accuracy:+.4f})")
        
        print("\n🔍 나이브 베이즈 모델 학습 중...")
        nb_acc = self.modeling_service.accuracy_by_naive_bayes(this)
        accuracy['나이브베이즈'] = nb_acc
        if nb_acc > baseline_accuracy:
            print(f"✅ 나이브 베이즈 정확도: {nb_acc:.4f} (베이스라인 대비: {nb_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ 나이브 베이즈 정확도: {nb_acc:.4f} (베이스라인 대비: {nb_acc - baseline_accuracy:+.4f})")
        
        print("\n🔍 K-최근접 이웃 모델 학습 중...")
        knn_acc = self.modeling_service.accuracy_by_knn(this)
        accuracy['KNN'] = knn_acc
        if knn_acc > baseline_accuracy:
            print(f"✅ KNN 정확도: {knn_acc:.4f} (베이스라인 대비: {knn_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ KNN 정확도: {knn_acc:.4f} (베이스라인 대비: {knn_acc - baseline_accuracy:+.4f})")
        
        print("\n🔍 서포트 벡터 머신 모델 학습 중...")
        svm_acc = self.modeling_service.accuracy_by_svm(this)
        accuracy['SVM'] = svm_acc
        if svm_acc > baseline_accuracy:
            print(f"✅ SVM 정확도: {svm_acc:.4f} (베이스라인 대비: {svm_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ SVM 정확도: {svm_acc:.4f} (베이스라인 대비: {svm_acc - baseline_accuracy:+.4f})")

        print("\n🔍 그레디언트 부스팅 모델 학습 중...")
        gb_acc = self.modeling_service.accuracy_by_gradient_boosting(this)
        accuracy['GradientBoosting'] = gb_acc
        if gb_acc > baseline_accuracy:
            print(f"✅ GradientBoosting 정확도: {gb_acc:.4f} (베이스라인 대비: {gb_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ GradientBoosting 정확도: {gb_acc:.4f} (베이스라인 대비: {gb_acc - baseline_accuracy:+.4f})")
        
        # 결과 저장
        this.accuracy = accuracy
        
        return accuracy

    def evaluation(self): #모델 평가
        """
        학습된 모델들의 정확도를 비교하여 최적의 모델 선택
        
        Returns:
            tuple: 가장 높은 정확도와 해당 모델 이름
        """
        print("\n" + "="*50)
        print("📊 타이타닉 생존자 예측 모델 평가 결과 📊".center(50))
        print("="*50)
        this = self.service.dataset
        
        if not hasattr(this, 'accuracy'):
            print("❌ 모델이 학습되지 않았습니다. 먼저 learning() 메소드를 실행하세요.")
            return None, None
        
        # 최고 성능 모델 찾기
        best_model = max(this.accuracy.items(), key=lambda x: x[1])
        
        print(f"\n🏆 최고 성능 모델: {best_model[0]}, 정확도: {best_model[1]:.4f}")
        print("\n📋 모델별 정확도 비교:")
        
        # 베이스라인과 비교한 성능 향상 출력
        baseline_accuracy = this.accuracy.get('베이스라인(로지스틱 회귀)', 0)
        
        # 정렬된 결과 표시
        sorted_results = sorted(this.accuracy.items(), key=lambda x: x[1], reverse=True)
        
        print("\n" + "-"*60)
        print(f"{'모델명':<20} {'정확도':<10} {'베이스라인 대비':<15} {'평가'}")
        print("-"*60)
        
        for i, (model, acc) in enumerate(sorted_results):
            if model == '베이스라인(로지스틱 회귀)':
                if i == 0:  # 베이스라인이 가장 좋은 경우
                    print(f"{model:<20} {acc:.4f}      {'---':<15} {'🏆 최고'}")
                else:
                    print(f"{model:<20} {acc:.4f}      {'---':<15} {'📊 기준'}")
            else:
                improvement = acc - baseline_accuracy
                status = ""
                if i == 0:
                    status = "🏆 최고"
                elif improvement > 0:
                    status = "✅ 개선"
                else:
                    status = "❌ 저조"
                    
                print(f"{model:<20} {acc:.4f}      {improvement:+.4f}        {status}")
        
        print("-"*60)
        
        this.best_model = best_model[0]
        this.best_accuracy = best_model[1]
        
        return best_model[1], best_model[0]
        
    def submit(self): #모델 배포
        """
        최적의 모델을 사용하여 테스트 데이터에 대한 예측 결과를 생성
        
        Returns:
            DataFrame: 예측 결과가 포함된 제출용 데이터프레임
        """
        
        print("\n" + "="*50)
        print("📊 타이타닉 생존자 예측 결과 제출 준비 📊".center(50))
        print("="*50)
        this = self.service.dataset
        
        if not hasattr(this, 'best_model'):
            print("❌ 모델 평가가 완료되지 않았습니다. 먼저 evaluation() 메소드를 실행하세요.")
            return None
        
        print(f"\n🏆 {this.best_model} 모델로 제출 파일 생성 (정확도: {this.best_accuracy:.4f})")
        
        # 최적의 모델(GradientBoosting) 생성 및 학습
        if this.best_model == 'GradientBoosting':
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            )
        else:
            # 다른 최적 모델이 선택된 경우 해당 모델 사용
            print(f"⚠️ {this.best_model} 모델은 아직 제출 파일 생성 기능이 구현되지 않았습니다.")
            print("⚠️ GradientBoosting 모델로 대체하여 제출 파일을 생성합니다.")
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            )
        
        # 모델 학습
        print("🔄 모델 학습 중...")
        model.fit(this.train, this.label)
        
        # 테스트 데이터로 예측
        print("🔄 테스트 데이터 예측 중...")
        predictions = model.predict(this.test)
        
        # 제출 파일 생성
        print("🔄 제출 파일 생성 중...")
        submission = pd.DataFrame({
            'PassengerId': this.id,
            'Survived': predictions
        })
        
        # Docker 환경에서 작동하는 절대 경로 사용
        # 애플리케이션 루트 디렉토리 기준 상대 경로
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_dir = os.path.join(base_dir, 'updated_data')
        
        # updated_data 폴더가 없으면 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일 저장
        submission_path = os.path.join(output_dir, 'submission.csv')
        submission.to_csv(submission_path, index=False)
        
        print(f"✅ 제출 파일이 생성되었습니다: {submission_path}")
        print(f"📊 총 {len(predictions)}개의 승객 데이터 예측 완료")
        print(f"   - 생존 예측 승객 수: {sum(predictions)} 명")
        print(f"   - 사망 예측 승객 수: {len(predictions) - sum(predictions)} 명")
        
        return submission

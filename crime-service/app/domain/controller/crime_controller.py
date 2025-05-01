from app.domain.service.crime_preprocessor import CrimePreprocessor
from app.domain.service.crime_visualizer import CrimeVisualizer
import app.domain.service.internal.correlate as correlation_analyzer 


class CrimeController:

    def __init__(self):
        self.preprocessor = CrimePreprocessor()
        self.visualizer = CrimeVisualizer()

    def preprocess(self, train, test): #데이터 전처리
        print("Controller: Calling preprocessor.preprocess...")
        processed_data = self.preprocessor.preprocess(train, test)
        print(f"Controller: Received processed data object from preprocessor: {processed_data}")
        return processed_data
    
    def correlation(self): #상관계수 분석
        print("Controller: Calling correlation_analyzer.load_and_analyze...") # 이 호출 방식 유지
        results = correlation_analyzer.load_and_analyze() # 이 호출 방식 유지
        print("Controller: Correlation analysis completed")
        # 결과 상태 확인 추가
        if isinstance(results, dict) and results.get('status') == 'Failure':
             print(f"Controller: Correlation analysis failed - {results.get('message', 'Unknown error')}")
        elif isinstance(results, dict) and results.get('status') == 'Partial Failure':
             print(f"Controller: Correlation analysis partially failed - {results.get('message', 'Unknown error')}")

        return results
    
    def get_correlation_results(self):
        """상관계수 분석 결과를 반환하는 함수"""
        return self.correlation()

    def draw_crime_map(self):
        """범죄 지도를 생성하는 함수"""
        print("Controller: Calling visualizer.draw_crime_map...")
        result = self.visualizer.draw_crime_map()
        if result.get("status") == "success":
            print(f"Controller: Crime map created successfully at {result.get('file_path')}")
        else:
            print(f"Controller: Failed to create crime map - {result.get('message')}")
        return result

    def learning(self): #모델 학습
        pass

    def evaluation(self): #모델 평가
        pass

    def deploy(self): #모델 배포
        pass


    

import re
from collections import Counter
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import matplotlib
from app.utils.logger import logger

# Matplotlib 백엔드를 Agg로 설정 (GUI 없는 환경)
matplotlib.use('Agg')

class SamsungReportAnalyzer:
    def __init__(self, report_path='./app/data/kr-Report_2018.txt', stopwords_path='./app/data/stopwords.txt', font_path='./app/data/D2Coding.ttf'):
        """
        분석기 초기화. 파일 경로와 폰트 경로를 설정합니다.
        """
        self.report_path = report_path
        self.stopwords_path = stopwords_path
        self.font_path = font_path
        self.okt = Okt()
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        """
        불용어 파일을 로드하여 set 형태로 반환합니다.
        """
        try:
            with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                stopwords_list = f.read().splitlines()
            # PRD 요구사항: 추가적인 불용어 (문맥상 불필요한 단어)
            additional_stopwords = {'삼성전자', '보고서'}
            return set(stopwords_list) | additional_stopwords
        except FileNotFoundError:
            logger.error(f"오류: 불용어 파일을 찾을 수 없습니다 - {self.stopwords_path}")
            return set() # 파일 없으면 빈 set 반환

    def load_report(self) -> str:
        """
        보고서 텍스트 파일을 로드합니다.
        """
        try:
            with open(self.report_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
            return report_text
        except FileNotFoundError:
            logger.error(f"오류: 보고서 파일을 찾을 수 없습니다 - {self.report_path}")
            return "" # 파일 없으면 빈 문자열 반환

    def clean_text(self, text: str) -> str:
        """
        텍스트에서 한글과 기본적인 공백 외의 문자를 제거합니다.
        """
        cleaned = re.sub(r'[^ ㄱ-힣]+', '', text) # 한글과 공백 제외하고 모두 제거
        cleaned = re.sub(r'\s+', ' ', cleaned).strip() # 다중 공백을 단일 공백으로, 양 끝 공백 제거
        return cleaned

    def extract_nouns(self, text: str) -> list[str]:
        """
        정제된 텍스트에서 명사를 추출합니다.
        """
        nouns = self.okt.nouns(text)
        return nouns

    def filter_keywords(self, nouns: list[str]) -> list[str]:
        """
        명사 리스트에서 불용어와 한 글자 단어를 제거합니다.
        """
        filtered = [
            noun for noun in nouns
            if noun not in self.stopwords and len(noun) > 1 # 불용어 아니고, 한 글자 이상
        ]
        return filtered

    def calculate_frequency(self, keywords: list[str], top_n: int = 100) -> dict[str, int]:
        """
        키워드 빈도를 계산하고 상위 N개를 반환합니다.
        """
        counter = Counter(keywords)
        most_common = counter.most_common(top_n)
        logger.info(f"상위 {top_n}개 키워드 빈도 계산 완료 (상위 5개: {most_common[:5]})")
        return dict(most_common)

    def generate_and_save_wordcloud(self, frequencies: dict[str, int], output_path: str) -> str | None:
        """
        키워드 빈도를 기반으로 워드 클라우드를 생성하고 이미지 파일로 저장한 후,
        성공 시 파일 경로를 반환하고 실패 시 None을 반환합니다.
        """
        # 출력 디렉토리 확인 및 생성 (필요한 경우)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"출력 디렉토리 생성: {output_dir}")
            except OSError as e:
                logger.error(f"출력 디렉토리 생성 실패: {e}")
                return None # 디렉토리 생성 실패 시 중단

        try:
            # 폰트 파일 존재 여부 확인
            if not os.path.exists(self.font_path):
                 logger.error(f"오류: 워드 클라우드 생성을 위한 폰트 파일을 찾을 수 없습니다 - {self.font_path}")
                 return None

            wc = WordCloud(font_path=self.font_path,
                           width=800, height=800,
                           background_color='white',
                           relative_scaling=0.2)

            wc.generate_from_frequencies(frequencies)

            # plt 객체를 직접 사용하지 않고 바로 파일로 저장
            wc.to_file(output_path)
            logger.info(f"워드 클라우드 이미지가 저장되었습니다: {output_path}")
            return output_path # 성공 시 저장된 파일 경로 반환

        except Exception as e:
            logger.error(f"워드 클라우드 생성 또는 저장 중 오류 발생: {e}")
            return None # 오류 발생 시 None 반환


    def process(self, top_n_keywords: int = 100, output_image_path: str = 'temp_wordcloud.png') -> str | None:
        """
        전체 분석 프로세스를 실행하고, 생성된 워드클라우드 이미지 파일 경로를 반환합니다.
        실패 시 None을 반환합니다.
        """
        logger.info("워드클라우드 분석 프로세스 시작...") # 프로세스 시작 로그 추가
        # 1. 데이터 로딩
        report_text = self.load_report()
        if not report_text: return None

        # 2. 텍스트 정제
        cleaned_text = self.clean_text(report_text)
        logger.debug("텍스트 정제 완료")

        # 3. 명사 추출
        nouns = self.extract_nouns(cleaned_text)
        logger.info(f"명사 추출 완료 (개수: {len(nouns)})")

        # 4. 불용어 제거
        keywords = self.filter_keywords(nouns)
        logger.info(f"불용어 제거 완료 (남은 키워드 수: {len(keywords)})")
        if not keywords:
            logger.warning("필터링 후 남은 키워드가 없습니다.")
            return None

        # 5. 빈도 계산
        frequencies = self.calculate_frequency(keywords, top_n=top_n_keywords)
        if not frequencies:
            logger.warning("빈도 계산 결과가 없습니다.")
            return None

        # 6. 워드 클라우드 생성 및 저장, 결과 경로 반환
        saved_path = self.generate_and_save_wordcloud(frequencies, output_path=output_image_path)
        if saved_path:
            logger.info("워드클라우드 분석 프로세스 완료.") # 성공 로그
        else:
            logger.error("워드클라우드 분석 프로세스 중 오류 발생.") # 실패 로그
        return saved_path
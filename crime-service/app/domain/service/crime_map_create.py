import os
import json
import pandas as pd
import numpy as np
import folium
from fastapi import HTTPException
import logging
import traceback

logger = logging.getLogger(__name__)

class CrimeMapCreator:
    def __init__(self, data_dir='app/updated_data', output_dir='app/saved_data'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        # 필요한 파일 경로 미리 정의
        self.police_norm_file = os.path.join(self.data_dir, 'police_norm_in_seoul.csv')
        self.crime_file = os.path.join(self.data_dir, 'crime_in_seoul.csv')
        self.geo_json_file = os.path.join(self.data_dir, 'geo_simple.json')
        self.police_pos_file = os.path.join(self.data_dir, 'police_pos.csv')
        self.output_map_file = os.path.join(self.output_dir, 'crime_map.html')

        # 디렉토리 생성 확인
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"데이터 디렉토리 확인: {self.data_dir}")
        logger.info(f"출력 디렉토리 확인: {self.output_dir}")

    def create_map(self) -> str:
        """범죄 지도를 생성하고 저장된 파일 경로를 반환합니다."""
        try:
            logger.info("범죄 지도 생성 시작...")
            police_norm, crime, state_geo, police_pos = self._load_required_data()
            folium_map = self._create_folium_map(police_norm, state_geo, police_pos)
            self._save_map_html(folium_map)
            logger.info(f"범죄 지도가 성공적으로 생성되었습니다: {self.output_map_file}")
            return self.output_map_file
        except FileNotFoundError as e:
            logger.error(f"필수 파일 로드 실패: {e}")
            raise HTTPException(status_code=404, detail=f"필수 데이터 파일을 찾을 수 없습니다: {e.filename}")
        except KeyError as e:
             logger.error(f"데이터 처리 중 필수 컬럼 부재: {e}")
             raise HTTPException(status_code=400, detail=f"데이터 처리 중 필요한 컬럼({e})이 없습니다.")
        except Exception as e:
            logger.error(f"지도 생성 중 예상치 못한 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            # 예외 발생 시 구체적인 타입과 메시지 포함
            raise HTTPException(status_code=500, detail=f"지도 생성 중 서버 오류 발생: {type(e).__name__} - {str(e)}")


    def _load_required_data(self):
        """지도 생성에 필요한 데이터를 로드합니다."""
        logger.info("필수 데이터 로드 중...")

        # police_norm 데이터 로드 (파일 없으면 기본 생성 로직은 제거하고 에러 발생)
        if not os.path.exists(self.police_norm_file):
            # logger.warning(f"경고: {self.police_norm_file} 파일이 없습니다. 기본 데이터를 생성합니다.")
            # police_norm = self._create_default_police_norm()
            # police_norm.to_csv(self.police_norm_file, index=False)
            raise FileNotFoundError(f"{self.police_norm_file} 파일을 찾을 수 없습니다.")
        police_norm = pd.read_csv(self.police_norm_file)
        logger.info(f"{self.police_norm_file} 파일 로드 완료")
        police_norm = self._preprocess_police_norm(police_norm) # 전처리 추가

        # crime 데이터 로드 (사용되지 않으면 로드 불필요, 현재는 미사용으로 보임)
        # if not os.path.exists(self.crime_file):
        #     raise FileNotFoundError(f"{self.crime_file} 파일을 찾을 수 없습니다.")
        # crime = pd.read_csv(self.crime_file)
        # logger.info(f"{self.crime_file} 파일 로드 완료")
        crime = None # 임시로 None 처리 (지도 생성에 직접 사용되지 않음)

        # GeoJSON 데이터 로드
        if not os.path.exists(self.geo_json_file):
            # logger.warning(f"경고: {self.geo_json_file} 파일이 없습니다. 기본 지리 정보를 생성합니다.")
            # state_geo = self._create_default_geojson()
            # try:
            #     with open(self.geo_json_file, 'w', encoding='utf-8') as f:
            #         json.dump(state_geo, f, ensure_ascii=False)
            #     logger.info(f"기본 지리 정보 파일이 생성되었습니다: {self.geo_json_file}")
            # except Exception as e:
            #     logger.error(f"지리 정보 파일 생성 실패: {str(e)}")
            #     # 파일 생성 실패 시 에러 발생
            #     raise IOError(f"기본 GeoJSON 파일 생성에 실패했습니다: {str(e)}")
            raise FileNotFoundError(f"{self.geo_json_file} 파일을 찾을 수 없습니다.")
        try:
             with open(self.geo_json_file, 'r', encoding='utf-8') as f:
                 state_geo = json.load(f)
             logger.info(f"{self.geo_json_file} 파일 로드 완료")
        except json.JSONDecodeError as e:
            logger.error(f"{self.geo_json_file} 파일 로드 중 JSON 디코딩 오류: {e}")
            raise ValueError(f"{self.geo_json_file} 파일의 형식이 올바르지 않습니다.")


        # 경찰서 위치 데이터 로드
        if not os.path.exists(self.police_pos_file):
            # logger.warning(f"경고: {self.police_pos_file} 파일이 없습니다. 기본 데이터를 생성합니다.")
            # police_pos = self._create_default_police_pos()
            # police_pos.to_csv(self.police_pos_file, index=False)
             raise FileNotFoundError(f"{self.police_pos_file} 파일을 찾을 수 없습니다.")
        police_pos = pd.read_csv(self.police_pos_file)
        logger.info(f"{self.police_pos_file} 파일 로드 완료")
        police_pos = self._preprocess_police_pos(police_pos) # 전처리 추가

        return police_norm, crime, state_geo, police_pos

    def _preprocess_police_norm(self, police_norm_df: pd.DataFrame) -> pd.DataFrame:
        """police_norm 데이터를 전처리합니다 (자치구 컬럼 확인 등)."""
        logger.info("police_norm 데이터 전처리 중...")
        # 자치구 컬럼명 통일 ('구별' -> '자치구')
        if '구별' in police_norm_df.columns and '자치구' not in police_norm_df.columns:
            police_norm_df['자치구'] = police_norm_df['구별']
            logger.info("컬럼명 변경: '구별' -> '자치구'")

        # 필수 컬럼 확인
        required_cols = ['자치구', '범죄'] # choropleth에 사용할 컬럼
        for col in required_cols:
            if col not in police_norm_df.columns:
                 # '범죄' 컬럼이 없으면 다른 이름 확인 (e.g., '범죄율')
                 if col == '범죄' and '범죄율' in police_norm_df.columns:
                      police_norm_df['범죄'] = police_norm_df['범죄율']
                      logger.info("컬럼명 변경: '범죄율' -> '범죄'")
                 else:
                      logger.error(f"police_norm 데이터에 필수 컬럼 '{col}'이 없습니다. 사용 가능한 컬럼: {police_norm_df.columns.tolist()}")
                      raise KeyError(f"police_norm 데이터에 필수 컬럼 '{col}'이 없습니다.")

        logger.info(f"police_norm 데이터 전처리 완료. 컬럼: {police_norm_df.columns.tolist()}")
        return police_norm_df

    def _preprocess_police_pos(self, police_pos_df: pd.DataFrame) -> pd.DataFrame:
        """police_pos 데이터를 전처리합니다 (검거율 계산 등)."""
        logger.info("police_pos 데이터 전처리 중...")
        # 필수 컬럼 확인
        required_cols = ['lat', 'lng'] # 마커 표시에 사용할 컬럼
        for col in required_cols:
             if col not in police_pos_df.columns:
                 logger.error(f"police_pos 데이터에 필수 컬럼 '{col}'이 없습니다. 사용 가능한 컬럼: {police_pos_df.columns.tolist()}")
                 raise KeyError(f"police_pos 데이터에 필수 컬럼 '{col}'이 없습니다.")

        # 검거율 컬럼 계산 ('검거' 컬럼이 없는 경우)
        if '검거' not in police_pos_df.columns:
            detection_cols = [col for col in police_pos_df.columns if '검거' in col and col.endswith(' 검거')]
            if detection_cols:
                logger.info(f"'검거' 컬럼이 없어 다음 컬럼들로 계산합니다: {detection_cols}")
                # 각 검거 건수를 최대값으로 나누어 정규화 후 합산 (기존 로직과 동일하게)
                # 0으로 나누는 경우 방지
                max_values = police_pos_df[detection_cols].max()
                # 최대값이 0인 컬럼은 1로 대체하여 0으로 나누는 것을 방지
                max_values[max_values == 0] = 1
                tmp = police_pos_df[detection_cols] / max_values
                police_pos_df['검거'] = np.sum(tmp, axis=1)
                logger.info("'검거' 컬럼 계산 완료.")
            else:
                logger.warning("경고: '검거' 컬럼 및 계산 가능한 'XX 검거' 컬럼이 없어 기본값 1을 사용합니다.")
                police_pos_df['검거'] = 1 # 기본값 할당

        logger.info(f"police_pos 데이터 전처리 완료. 컬럼: {police_pos_df.columns.tolist()}")
        return police_pos_df


    def _create_folium_map(self, police_norm, state_geo, police_pos):
        """Folium을 사용하여 지도를 생성합니다."""
        logger.info("Folium 지도 생성 중...")
        # 기본 지도 생성 (서울 중심)
        folium_map = folium.Map(location=[37.5502, 126.982], zoom_start=12, tiles='OpenStreetMap')

        # 1. Choropleth (구별 범죄율)
        try:
            logger.info("Choropleth 레이어 추가 중 (구별 범죄율)...")
            folium.Choropleth(
                geo_data=state_geo,
                data=police_norm, # 데이터프레임 직접 전달
                columns=['자치구', '범죄'], # 사용할 컬럼 지정
                key_on='feature.id', # GeoJSON의 id와 매칭될 컬럼
                fill_color='PuRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='범죄 발생률 (정규화)',
                reset=True,
            ).add_to(folium_map)
            logger.info("Choropleth 레이어 추가 완료.")
        except KeyError as e:
             logger.error(f"Choropleth 생성 중 오류: 필요한 컬럼({e})이 police_norm 데이터에 없습니다.")
             raise # 오류를 다시 발생시켜 상위에서 처리하도록 함
        except Exception as e:
            logger.error(f"Choropleth 생성 중 예상치 못한 오류: {str(e)}")
            logger.error(traceback.format_exc())
            # Choropleth 실패해도 지도 자체는 반환될 수 있도록 경고만 로깅 (선택사항)
            # raise # 또는 여기서도 에러 발생시켜 전체 실패 처리


        # 2. CircleMarker (경찰서 위치 및 검거율)
        try:
            logger.info("CircleMarker 레이어 추가 중 (경찰서 위치 및 검거율)...")
            for idx in police_pos.index:
                 lat = police_pos.loc[idx, 'lat']
                 lng = police_pos.loc[idx, 'lng']
                 radius = police_pos.loc[idx, '검거'] * 10  # 검거율에 비례한 크기
                 popup_text = f"검거율: {police_pos.loc[idx, '검거']:.2f}"

                 folium.CircleMarker(
                     location=[lat, lng],
                     radius=max(radius, 3), # 최소 반경 보장
                     fill=True,
                     fill_color='#0a0a32', # 남색 계열
                     fill_opacity=0.6,
                     color=None, # 테두리 없음
                     popup=popup_text
                 ).add_to(folium_map)
            logger.info("CircleMarker 레이어 추가 완료.")
        except KeyError as e:
             logger.error(f"CircleMarker 생성 중 오류: 필요한 컬럼({e})이 police_pos 데이터에 없습니다.")
             raise
        except Exception as e:
            logger.error(f"CircleMarker 생성 중 예상치 못한 오류: {str(e)}")
            logger.error(traceback.format_exc())
            # 마커 생성 실패해도 지도 자체는 반환될 수 있도록 경고만 로깅 (선택사항)
            # raise


        # 레이어 컨트롤 추가 (선택사항)
        folium.LayerControl().add_to(folium_map)

        logger.info("Folium 지도 생성 완료.")
        return folium_map

    def _save_map_html(self, folium_map):
        """생성된 Folium 지도를 HTML 파일로 저장합니다."""
        try:
            logger.info(f"생성된 지도를 HTML 파일로 저장 중: {self.output_map_file}")
            folium_map.save(self.output_map_file)
            logger.info("지도 저장 완료.")
        except Exception as e:
            logger.error(f"지도 저장 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise IOError(f"지도를 HTML 파일로 저장하는 데 실패했습니다: {str(e)}")

    # --- 기본 데이터 생성 함수 (참고용, 실제 사용 시 파일 부재는 에러 처리) ---
    # def _create_default_police_norm(self):
    #     # ... (기존 코드 참고하여 기본 데이터프레임 생성)
    #     pass
    #
    # def _create_default_geojson(self):
    #     # ... (기존 코드 참고하여 기본 GeoJSON 생성)
    #     pass
    #
    # def _create_default_police_pos(self):
    #     # ... (기존 코드 참고하여 기본 데이터프레임 생성)
    #     pass

# 사용 예시 (테스트용)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     creator = CrimeMapCreator()
#     try:
#         map_file_path = creator.create_map()
#         print(f"지도 생성 완료: {map_file_path}")
#     except HTTPException as e:
#         print(f"지도 생성 실패 (HTTPException): {e.status_code} - {e.detail}")
#     except Exception as e:
#         print(f"지도 생성 실패 (일반 오류): {e}")



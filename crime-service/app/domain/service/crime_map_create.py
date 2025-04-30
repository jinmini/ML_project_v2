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
    def __init__(self, data_dir='app/up_data', output_dir='app/map_data'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        # 필요한 파일 경로 미리 정의
        self.police_norm_file = os.path.join(self.data_dir, 'police_norm_in_seoul.csv')
        self.geo_json_file = os.path.join(self.data_dir, 'geo_simple.json')
        self.output_map_file = os.path.join(self.output_dir, 'crime_map.html')

        # 디렉토리 생성 확인
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"데이터 읽기 디렉토리 확인: {self.data_dir}")
        logger.info(f"지도 출력 디렉토리 확인: {self.output_dir}")

    def create_map(self) -> str:
        """범죄 지도를 생성하고 저장된 파일 경로를 반환합니다."""
        try:
            logger.info("범죄 지도 생성 시작...")
            police_norm, state_geo = self._load_required_data()
            folium_map = self._create_folium_map(police_norm, state_geo)
            self._save_map_html(folium_map)
            logger.info(f"범죄 지도가 성공적으로 생성되었습니다: {self.output_map_file}")
            return self.output_map_file
        except FileNotFoundError as e:
            logger.error(f"필수 파일 로드 실패: {e}")
            # FileNotFoundError 객체에는 filename 속성이 없을 수 있음
            raise HTTPException(status_code=404, detail=f"필수 데이터 파일을 찾을 수 없습니다: {e}")
        except KeyError as e:
             logger.error(f"데이터 처리 중 필수 컬럼 부재: {e}")
             raise HTTPException(status_code=400, detail=f"데이터 처리 중 필요한 컬럼({e})이 없습니다.")
        except ValueError as e:
             logger.error(f"데이터 처리 중 값 또는 형식 오류: {e}")
             raise HTTPException(status_code=400, detail=f"데이터 처리 중 오류 발생: {e}")
        except Exception as e:
            logger.error(f"지도 생성 중 예상치 못한 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"지도 생성 중 서버 오류 발생: {type(e).__name__} - {str(e)}")

    def _load_required_data(self):
        """지도 생성에 필요한 데이터를 로드하고 기본적인 핸들링을 수행합니다."""
        logger.info("필수 데이터 로드 중...")

        # police_norm 데이터 로드
        if not os.path.exists(self.police_norm_file):
            raise FileNotFoundError(self.police_norm_file)
        try:
            police_norm = pd.read_csv(self.police_norm_file)
            logger.info(f"{self.police_norm_file} 파일 로드 완료")
            # '자치구' 컬럼 임시 처리 (Workaround) - 근본 원인 해결 후 제거 고려
            if '자치구' not in police_norm.columns and 'Unnamed: 0' in police_norm.columns:
                logger.warning(f"'{self.police_norm_file}' 파일에 '자치구' 컬럼이 없습니다. 'Unnamed: 0' 컬럼을 '자치구'로 사용합니다. (파일 재생성 권장)")
                police_norm.rename(columns={'Unnamed: 0': '자치구'}, inplace=True)
                if not pd.api.types.is_string_dtype(police_norm['자치구']):
                     logger.error(f"'{self.police_norm_file}' 파일의 'Unnamed: 0' 컬럼이 자치구 이름이 아닙니다. 지도 생성 불가.")
                     raise ValueError(f"'{self.police_norm_file}'에서 유효한 '자치구' 정보를 찾을 수 없습니다.")
        except Exception as e:
            logger.error(f"{self.police_norm_file} 파일 처리 중 오류: {e}")
            raise ValueError(f"{self.police_norm_file} 파일을 처리하는 중 오류가 발생했습니다: {e}")

        police_norm = self._preprocess_police_norm(police_norm)

        # GeoJSON 데이터 로드
        if not os.path.exists(self.geo_json_file):
            raise FileNotFoundError(self.geo_json_file)
        try:
            with open(self.geo_json_file, 'r', encoding='utf-8') as f:
                state_geo = json.load(f)
            logger.info(f"{self.geo_json_file} 파일 로드 완료")
        except json.JSONDecodeError as e:
            logger.error(f"{self.geo_json_file} 파일 로드 중 JSON 디코딩 오류: {e}")
            raise ValueError(f"{self.geo_json_file} 파일의 형식이 올바르지 않습니다.")
        except Exception as e:
            logger.error(f"{self.geo_json_file} 파일 처리 중 오류: {e}")
            raise ValueError(f"{self.geo_json_file} 파일을 처리하는 중 오류가 발생했습니다: {e}")

        return police_norm, state_geo

    def _preprocess_police_norm(self, police_norm_df: pd.DataFrame) -> pd.DataFrame:
        """police_norm 데이터를 전처리합니다 (컬럼 존재 여부 확인 등)."""
        logger.info("police_norm 데이터 전처리 중...")
        # 필수 컬럼 존재 여부 최종 확인
        required_cols = ['자치구', '범죄'] # choropleth에 사용할 컬럼
        for col in required_cols:
            if col not in police_norm_df.columns:
                if col == '범죄' and '범죄율' in police_norm_df.columns:
                     police_norm_df['범죄'] = police_norm_df['범죄율']
                     logger.info("컬럼명 변경 시도: '범죄율' -> '범죄'")
                else:
                     logger.error(f"police_norm 데이터에 필수 컬럼 '{col}'이 최종적으로 없습니다. 사용 가능한 컬럼: {police_norm_df.columns.tolist()}")
                     raise KeyError(f"police_norm 데이터에 필수 컬럼 '{col}'이 없습니다.")

        logger.info(f"police_norm 데이터 전처리 완료. 컬럼: {police_norm_df.columns.tolist()}")
        return police_norm_df

    def _create_folium_map(self, police_norm, state_geo):
        """Folium을 사용하여 지도를 생성합니다 (Choropleth + 위험 지역 마커 포함)."""
        logger.info("Folium 지도 생성 중... (Choropleth + Markers)")
        # 기본 지도 생성 (서울 중심)
        folium_map = folium.Map(location=[37.5502, 126.982], zoom_start=12, tiles='OpenStreetMap')

        # 1. Choropleth (구별 범죄율)
        try:
            logger.info("Choropleth 레이어 추가 중 (구별 범죄율)...")
            if not pd.api.types.is_string_dtype(police_norm['자치구']):
                 logger.warning("'자치구' 컬럼 타입이 문자열이 아닙니다. Choropleth 매칭 실패 가능성이 있습니다.")

            folium.Choropleth(
                geo_data=state_geo,
                data=police_norm,
                columns=['자치구', '범죄'],
                key_on='feature.id',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.3,
                legend_name='자치구별 범죄 지수 (높을수록 붉은색)',
                reset=True,
                name='자치구별 범죄 지수' # 레이어 컨트롤에 표시될 이름
            ).add_to(folium_map)
            logger.info("Choropleth 레이어 추가 완료.")
        except KeyError as e:
             logger.error(f"Choropleth 생성 중 오류: 필요한 컬럼({e})이 police_norm 데이터에 없습니다.")
             raise
        except Exception as e:
            logger.error(f"Choropleth 생성 중 예상치 못한 오류: {str(e)}")
            logger.error(traceback.format_exc())
            # raise

        # 2. 위험 지역 마커 추가 (상위 3개 구)
        try:
             logger.info("위험 지역 마커 추가 중 (상위 3개 구)...")
             # 범죄 지수 기준 상위 3개 구 선정
             top3_districts = police_norm.nlargest(3, '범죄')
             logger.info(f"상위 3개 위험 지역: {top3_districts['자치구'].tolist()}")

             # 마커를 담을 FeatureGroup 생성
             marker_group = folium.FeatureGroup(name='위험 지역 (범죄 상위 3)', show=True) # 기본적으로 보이도록 설정

             # GeoJSON에서 상위 3개 구의 좌표 찾아서 마커 추가
             for idx, row in top3_districts.iterrows():
                 gu_name = row['자치구']
                 crime_value = row['범죄']
                 found_feature = False
                 for feature in state_geo['features']:
                     if feature['id'] == gu_name:
                         # 폴리곤의 첫 번째 좌표를 대표 위치로 사용 (간단한 방법)
                         # geometry 타입 확인 (Polygon 가정)
                         coords = feature['geometry']['coordinates'][0]
                         if coords and len(coords) > 0:
                             # GeoJSON 좌표는 (경도, 위도) 순서일 수 있으므로 확인 필요
                             # Folium은 (위도, 경도) 순서를 사용
                             lng, lat = coords[0]
                             marker_location = [lat, lng]

                             # 마커 생성
                             folium.Marker(
                                 location=marker_location,
                                 # 툴팁: 마우스 오버 시 표시
                                 tooltip=f"{gu_name}: 범죄 지수 {crime_value:.2f}",
                                 # 아이콘 설정
                                 icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa') # Font Awesome 아이콘 사용
                             ).add_to(marker_group)
                             found_feature = True
                             break # 해당 구 찾았으므로 내부 루프 종료
                 if not found_feature:
                     logger.warning(f"상위 3개 구 '{gu_name}'에 해당하는 좌표를 GeoJSON에서 찾지 못했습니다.")

             marker_group.add_to(folium_map) # 마커 그룹을 지도에 추가
             logger.info("위험 지역 마커 추가 완료.")

        except KeyError as e:
            logger.error(f"위험 지역 마커 생성 중 오류: 필요한 컬럼({e})이 police_norm 데이터에 없습니다.")
            # 마커 추가 실패해도 지도 생성은 계속 진행
        except Exception as e:
            logger.error(f"위험 지역 마커 생성 중 예상치 못한 오류: {str(e)}")
            logger.error(traceback.format_exc())
            # 마커 추가 실패해도 지도 생성은 계속 진행

        # 레이어 컨트롤 추가 (Choropleth와 Marker 그룹 제어)
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



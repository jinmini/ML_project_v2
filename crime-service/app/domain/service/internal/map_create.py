import os
import json
import pandas as pd
import numpy as np
import folium
from fastapi import HTTPException
import logging
import traceback
from folium.plugins import MarkerCluster

logger = logging.getLogger(__name__)

def load_required_data(data_dir: str, 
                       police_norm_file: str, 
                       geo_json_file: str, 
                       cctv_file: str,
                       pop_file: str):
    """지도 생성에 필요한 데이터를 로드하고 기본적인 핸들링을 수행합니다."""
    logger.info("필수 데이터 로드 중 (Police Norm, GeoJSON, CCTV, Population)...")

    police_norm_path = os.path.join(data_dir, police_norm_file)
    geo_json_path = os.path.join(data_dir, geo_json_file)
    cctv_path = os.path.join(data_dir, cctv_file)
    pop_path = os.path.join(data_dir, pop_file)

    loaded_data = {}

    def _load_csv(file_path, name):
        if not os.path.exists(file_path):
            logger.error(f"필수 파일 없음: {file_path}")
            raise FileNotFoundError(file_path)
        try:
            df = pd.read_csv(file_path)
            logger.info(f"{file_path} ({name}) 파일 로드 완료 (shape: {df.shape})")
            loaded_data[name] = df
        except Exception as e:
            logger.error(f"{file_path} ({name}) 파일 처리 중 오류: {e}")
            raise ValueError(f"{file_path} ({name}) 파일 처리 오류: {e}")

    _load_csv(police_norm_path, 'police_norm')
    _load_csv(cctv_path, 'cctv')
    _load_csv(pop_path, 'pop')

    if not os.path.exists(geo_json_path):
        logger.error(f"필수 파일 없음: {geo_json_path}")
        raise FileNotFoundError(geo_json_path)
    try:
        with open(geo_json_path, 'r', encoding='utf-8') as f:
            state_geo = json.load(f)
        logger.info(f"{geo_json_path} (GeoJSON) 파일 로드 완료")
        loaded_data['state_geo'] = state_geo
    except json.JSONDecodeError as e:
        logger.error(f"{geo_json_path} 파일 JSON 디코딩 오류: {e}")
        raise ValueError(f"{geo_json_path} 파일 형식 오류.")
    except Exception as e:
        logger.error(f"{geo_json_path} 파일 처리 중 오류: {e}")
        raise ValueError(f"{geo_json_path} 파일 처리 오류: {e}")

    police_norm = preprocess_police_norm(loaded_data['police_norm'])
    cctv_data = preprocess_cctv_data(loaded_data['cctv'])
    pop_data = preprocess_pop_data(loaded_data['pop'])

    return police_norm, loaded_data['state_geo'], cctv_data, pop_data

def preprocess_cctv_data(cctv_df: pd.DataFrame) -> pd.DataFrame:
    """CCTV 데이터를 전처리합니다."""
    logger.info("CCTV 데이터 전처리 중...")
    if '기관명' in cctv_df.columns and '자치구' not in cctv_df.columns:
        cctv_df = cctv_df.rename(columns={'기관명': '자치구'})
        logger.info("CCTV 데이터: '기관명' -> '자치구' 컬럼명 변경")
    
    if '소계' not in cctv_df.columns:
        logger.error("CCTV 데이터에 '소계' 컬럼이 없습니다.")
        raise KeyError("CCTV 데이터에 '소계' 컬럼이 없습니다.")
    if not pd.api.types.is_numeric_dtype(cctv_df['소계']):
        logger.warning("CCTV 데이터 '소계' 컬럼 숫자형 변환 시도")
        cctv_df['소계'] = pd.to_numeric(cctv_df['소계'], errors='coerce')
        if cctv_df['소계'].isnull().any():
             logger.error("CCTV 데이터 '소계' 컬럼 숫자형 변환 실패. NaN 포함")
             cctv_df['소계'] = cctv_df['소계'].fillna(0)
    
    if '자치구' in cctv_df.columns:
        cctv_df = cctv_df[['자치구', '소계']].copy()
        logger.info(f"CCTV 데이터 전처리 완료. 최종 컬럼: {cctv_df.columns.tolist()}")
        return cctv_df
    else:
        logger.error("CCTV 데이터에 '자치구' 컬럼이 최종적으로 없습니다.")
        raise KeyError("CCTV 데이터에 '자치구' 컬럼이 없습니다.")

def preprocess_pop_data(pop_df: pd.DataFrame) -> pd.DataFrame:
    """인구 데이터를 전처리합니다."""
    logger.info("인구 데이터 전처리 중...")
    if '구별' in pop_df.columns and '자치구' not in pop_df.columns:
        pop_df = pop_df.rename(columns={'구별': '자치구'})
        logger.info("인구 데이터: '구별' -> '자치구' 컬럼명 변경")
    elif '자치구' not in pop_df.columns:
        if len(pop_df.columns) > 0:
             first_col = pop_df.columns[0]
             pop_df = pop_df.rename(columns={first_col: '자치구'})
             logger.warning(f"인구 데이터: '자치구' 컬럼 없어 첫 컬럼('{first_col}') 사용")
        else:
             logger.error("인구 데이터에 '자치구' 또는 식별 가능한 컬럼 없음")
             raise KeyError("인구 데이터에 자치구 컬럼 필요")
    
    rename_map = {}
    if len(pop_df.columns) > 1 and '인구수' not in pop_df.columns: rename_map[pop_df.columns[1]] = '인구수'
    if len(pop_df.columns) > 2 and '한국인' not in pop_df.columns: rename_map[pop_df.columns[2]] = '한국인'
    if len(pop_df.columns) > 3 and '외국인' not in pop_df.columns: rename_map[pop_df.columns[3]] = '외국인'
    if len(pop_df.columns) > 4 and '고령자' not in pop_df.columns: rename_map[pop_df.columns[4]] = '고령자'
    if rename_map:
        pop_df = pop_df.rename(columns=rename_map)
        logger.info(f"인구 데이터 컬럼명 변경: {rename_map}")

    if '합계' in pop_df['자치구'].values:
        pop_df = pop_df[pop_df['자치구'] != '합계'].copy()
        logger.info("인구 데이터: '합계' 행 제거")

    if '인구수' not in pop_df.columns:
         logger.error("인구 데이터에 '인구수' 컬럼이 없습니다.")
         raise KeyError("인구 데이터에 '인구수' 컬럼 필요")
    if not pd.api.types.is_numeric_dtype(pop_df['인구수']):
         pop_df['인구수'] = pd.to_numeric(pop_df['인구수'], errors='coerce')
    pop_df['인구수'] = pop_df['인구수'].fillna(0)

    pop_df = pop_df[['자치구', '인구수']].copy()
    logger.info(f"인구 데이터 전처리 완료. 최종 컬럼: {pop_df.columns.tolist()}")
    return pop_df

def preprocess_police_norm(police_norm_df: pd.DataFrame) -> pd.DataFrame:
    """police_norm 데이터를 전처리합니다 (컬럼 존재 여부 확인 등)."""
    logger.info("police_norm 데이터 전처리 중...")
    if '자치구' not in police_norm_df.columns and 'Unnamed: 0' in police_norm_df.columns:
        logger.warning(f"'{police_norm_df}' 파일에 '자치구' 컬럼 없음. 'Unnamed: 0' 사용 시도.")
        police_norm_df.rename(columns={'Unnamed: 0': '자치구'}, inplace=True)
        if not pd.api.types.is_string_dtype(police_norm_df['자치구']):
                logger.error(f"'{police_norm_df}' 파일의 'Unnamed: 0' 컬럼이 유효한 자치구 이름 아님.")
                raise ValueError(f"'{police_norm_df}'에서 유효한 '자치구' 정보 찾을 수 없음.")
    elif '자치구' not in police_norm_df.columns:
         logger.error("police_norm 데이터에 '자치구' 컬럼 없음")
         raise KeyError("police_norm 데이터에 '자치구' 컬럼 필요")

    required_cols = ['자치구', '범죄', '검거']
    for col in required_cols:
        if col not in police_norm_df.columns:
            if col == '범죄' and '범죄율' in police_norm_df.columns:
                 police_norm_df['범죄'] = police_norm_df['범죄율']
                 logger.info("컬럼명 변경: '범죄율' -> '범죄'")
            elif col != '검거':
                 logger.error(f"police_norm 데이터 필수 컬럼 '{col}' 없음. 사용 가능: {police_norm_df.columns.tolist()}")
                 raise KeyError(f"police_norm 데이터 필수 컬럼 '{col}' 없음.")
    
    for col in ['범죄', '검거']:
        if col in police_norm_df.columns:
            if not pd.api.types.is_numeric_dtype(police_norm_df[col]):
                police_norm_df[col] = pd.to_numeric(police_norm_df[col], errors='coerce').fillna(0)
                logger.info(f"'{col}' 컬럼 숫자형 변환 및 NaN 처리 완료")
        else:
            logger.warning(f"police_norm 데이터에 '{col}' 컬럼 없어 처리 건너뜀 (툴팁에 영향)")

    logger.info(f"police_norm 데이터 전처리 완료. 컬럼: {police_norm_df.columns.tolist()}")
    return police_norm_df

def create_folium_map(police_norm: pd.DataFrame, 
                       state_geo: dict, 
                       cctv_data: pd.DataFrame, 
                       pop_data: pd.DataFrame):
    """Folium을 사용하여 지도를 생성합니다 (Choropleth + 필요지수 기반 마커).
    
    - Choropleth: 구별 범죄 지수
    - Markers:
        - 상위 5개 구 (설치 필요 지수 기준): 빨간색 아이콘 마커
        - 나머지 구: 설치 필요 지수 기반 크기 조절된 원형 마커
    """
    logger.info("Folium 지도 생성 시작... (Choropleth + Priority Markers)")
    
    # 1. 데이터 준비 및 병합
    try:
        merged_data = pd.merge(police_norm, cctv_data, on='자치구', how='left')
        merged_data = pd.merge(merged_data, pop_data, on='자치구', how='left')
        logger.info(f"지도용 데이터 병합 완료 (shape: {merged_data.shape})")

        # 필요한 컬럼 존재 확인 및 NaN 처리
        required_cols = {'범죄': 0.0, '검거': 0.0, '소계': 0.0, '인구수': 0.0}
        for col, default_val in required_cols.items():
            if col not in merged_data.columns:
                logger.warning(f"병합 데이터에 필수 컬럼 '{col}' 없음. 기본값({default_val}) 사용.")
                merged_data[col] = default_val
            merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce').fillna(default_val)
            logger.info(f"컬럼 '{col}' 숫자형 변환 및 NaN 처리 완료.")

        # CCTV 설치율 계산 (인구 만 명당)
        merged_data['cctv_per_10k_pop'] = np.where(
            merged_data['인구수'] > 0, 
            (merged_data['소계'] / merged_data['인구수']) * 10000, 
            0
        )
        merged_data['cctv_per_10k_pop'] = merged_data['cctv_per_10k_pop'].fillna(0)
        logger.info("CCTV 설치율 (인구 만 명당) 계산 완료")

        # CCTV 설치 필요 지수 계산 (예: 범죄율 / (CCTV설치율 + 1) )
        # 범죄 지수와 CCTV 설치율 값이 필요
        merged_data['CCTV_설치필요지수'] = np.where(
            merged_data['cctv_per_10k_pop'] >= 0, # 분모 0 방지 (+1 했으므로 항상 > 0)
            merged_data['범죄'] / (merged_data['cctv_per_10k_pop'] + 1),
            merged_data['범죄'] # 설치율이 NaN 등 이상하면 범죄율 자체를 지수로 사용 (대체 로직)
        )
        merged_data['CCTV_설치필요지수'] = merged_data['CCTV_설치필요지수'].fillna(0) # 계산 후 NaN 처리
        logger.info("CCTV 설치 필요 지수 계산 완료")

    except KeyError as e:
        logger.error(f"데이터 준비 중 필수 컬럼({e}) 없음")
        raise ValueError(f"지도 데이터 준비 중 오류: 컬럼 '{e}' 없음")
    except Exception as e:
        logger.error(f"데이터 준비 중 오류: {e}")
        logger.debug(traceback.format_exc())
        raise ValueError(f"지도 데이터 준비 중 오류: {e}")

    # GeoJSON properties에 데이터 병합 (Tooltip 표시용)
    logger.info("GeoJSON properties에 추가 데이터 병합 중...")
    try:
        merged_data_indexed = merged_data.set_index('자치구')
        for feature in state_geo['features']:
            gu_name = feature['id']
            if gu_name in merged_data_indexed.index:
                data_row = merged_data_indexed.loc[gu_name]
                feature['properties']['범죄'] = data_row.get('범죄', None)
                feature['properties']['검거'] = data_row.get('검거', None)
                feature['properties']['소계'] = data_row.get('소계', None)
                feature['properties']['인구수'] = data_row.get('인구수', None)
                feature['properties']['cctv_per_10k_pop'] = data_row.get('cctv_per_10k_pop', None)
                feature['properties']['CCTV_설치필요지수'] = data_row.get('CCTV_설치필요지수', None)
            else:
                logger.warning(f"GeoJSON의 '{gu_name}'에 해당하는 데이터를 merged_data에서 찾을 수 없음.")
                for col in ['범죄', '검거', '소계', '인구수', 'cctv_per_10k_pop', 'CCTV_설치필요지수']:
                    feature['properties'][col] = None
        logger.info("GeoJSON properties 데이터 병합 완료.")
    except Exception as e:
        logger.error(f"GeoJSON properties 병합 중 오류: {e}")
        logger.debug(traceback.format_exc())
        # 실패해도 계속 진행 시도

    # 2. 지도 객체 생성
    seoul_center = [37.5665, 126.9780] # 서울 중심 좌표
    folium_map = folium.Map(location=seoul_center, zoom_start=11, tiles='OpenStreetMap')

    # 3. Choropleth 레이어 추가 (범죄 지수 기준)
    try:
        logger.info("Choropleth 레이어 추가 중 (범죄 지수 기반)...")
        folium.Choropleth(
            geo_data=state_geo,             # 데이터 병합된 GeoJSON
            data=merged_data,               # 스타일링 및 범례 위한 DataFrame
            columns=['자치구', '범죄'],    # DataFrame 기준 key/value 컬럼
            key_on='feature.id',          # GeoJSON key ('강남구')
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.3,
            legend_name='자치구별 범죄 지수 (높을수록 붉은색)',
            name='자치구별 범죄 지수'
        ).add_to(folium_map).geojson.add_child(
            folium.features.GeoJsonTooltip(
                fields=['name', '범죄', '검거', '소계', '인구수', 'cctv_per_10k_pop', 'CCTV_설치필요지수'], 
                aliases=['자치구:', '범죄지수:', '검거지수:', 'CCTV수:', '인구수:', 'CCTV설치율(만명당):', '설치필요지수:'],
                localize=True,
                sticky=False,
                labels=True,
                style="""
                    background-color: #F0EFEF;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=800,
                # 숫자에 대해 소수점 포매팅 추가
                fmt = {
                     '범죄': '{:.2f}'.format,
                     '검거': '{:.2f}'.format, 
                     '소계': '{:,}'.format, 
                     '인구수': '{:,}'.format, 
                     'cctv_per_10k_pop': '{:.2f}'.format,
                     'CCTV_설치필요지수': '{:.2f}'.format
                }
            )
        )
        logger.info("Choropleth 레이어 및 툴팁 추가 완료.")
    except Exception as e:
        logger.error(f"Choropleth 생성 중 예상치 못한 오류: {str(e)}")
        logger.debug(traceback.format_exc())
        # 실패 시에도 마커 생성 시도

    # 4. 마커 추가 (설치 필요 지수 기준)
    try:
        logger.info("설치 필요 지수 기반 마커 추가 중...")
        marker_group = folium.FeatureGroup(name='구별 CCTV 설치 필요성', show=True).add_to(folium_map)

        # 상위 5개 구 선정
        top5_gu = merged_data.nlargest(5, 'CCTV_설치필요지수')
        top5_names = top5_gu['자치구'].tolist()
        logger.info(f"CCTV 설치 필요 상위 5개 구 선정: {top5_names}")

        # 마커 크기 스케일링 (나머지 구 대상 CircleMarker)
        min_size, max_size = 5, 15 # 원형 마커 최소/최대 크기
        priority_values = merged_data['CCTV_설치필요지수']
        # 0으로 나누는 경우 방지, 지수가 모두 같을 경우 대비
        min_val = priority_values.min()
        max_val = priority_values.max()
        if max_val == min_val:
             scaled_sizes = pd.Series([min_size] * len(merged_data), index=merged_data.index)
             logger.warning("모든 구의 설치 필요 지수가 동일하여 최소 크기로 고정.")
        else:
             scaled_sizes = ((priority_values - min_val) / (max_val - min_val)) * (max_size - min_size) + min_size
        
        # 각 구별 마커 추가
        for feature in state_geo['features']:
            gu_name = feature['id']
            properties = feature['properties']
            
            # 대표 좌표 찾기
            marker_location = None
            try:
                geom_type = feature['geometry']['type']
                coords = feature['geometry']['coordinates']
                if geom_type == 'Polygon':
                    # 폴리곤의 첫 번째 좌표 사용 (단순화)
                    representative_coord = coords[0][0] 
                elif geom_type == 'MultiPolygon':
                     # 가장 큰 폴리곤의 첫 번째 좌표 사용 (단순화)
                    representative_coord = coords[0][0][0]
                else:
                     logger.warning(f"'{gu_name}'의 geometry 타입({geom_type}) 처리 불가, 마커 스킵")
                     continue
                lng, lat = representative_coord
                marker_location = [lat, lng]
            except Exception as coord_err:
                logger.warning(f"'{gu_name}' 좌표 처리 중 오류 ({coord_err}), 마커 스킵")
                continue
                
            if marker_location is None: continue

            # 툴팁 내용 생성 (NaN 값 처리)
            crime_str = f"{properties.get('범죄', 0):.2f}" if pd.notna(properties.get('범죄')) else 'N/A'
            arrest_str = f"{properties.get('검거', 0):.1f}%" if pd.notna(properties.get('검거')) else 'N/A' # 검거율은 %로 표시 가정
            cctv_str = f"{int(properties.get('소계', 0)):,}" if pd.notna(properties.get('소계')) else 'N/A'
            pop_str = f"{int(properties.get('인구수', 0)):,}" if pd.notna(properties.get('인구수')) else 'N/A'
            cctv_rate_str = f"{properties.get('cctv_per_10k_pop', 0):.2f}" if pd.notna(properties.get('cctv_per_10k_pop')) else 'N/A'
            priority_str = f"{properties.get('CCTV_설치필요지수', 0):.2f}" if pd.notna(properties.get('CCTV_설치필요지수')) else 'N/A'
            
            tooltip_html = f"<b>{gu_name}</b><br>"
            if gu_name in top5_names: tooltip_html = f"<b>🔴 {gu_name}</b><br>"
            tooltip_html += (f"범죄지수: {crime_str}<br>"
                             f"검거지수: {arrest_str}<br>"
                             f"CCTV수: {cctv_str}<br>"
                             f"인구수: {pop_str}<br>"
                             f"CCTV설치율(만명당): {cctv_rate_str}<br>"
                             f"설치필요지수: {priority_str}")
            
            tooltip = folium.Tooltip(tooltip_html)

            # 마커 타입 결정 및 추가
            if gu_name in top5_names:
                # 상위 5개 구: 아이콘 마커
                folium.Marker(
                    location=marker_location,
                    icon=folium.Icon(color='red', icon='exclamation-sign'),
                    tooltip=tooltip
                ).add_to(marker_group)
            else:
                # 나머지 구: 크기 조절된 원형 마커
                # 해당 구의 스케일된 크기 가져오기
                radius = min_size # 기본값
                try:
                    if gu_name in merged_data_indexed.index:
                         radius = scaled_sizes.loc[gu_name]
                except KeyError:
                     logger.warning(f"'{gu_name}'의 스케일된 마커 크기 찾기 실패, 기본 크기 사용")
                except Exception as scale_err:
                    logger.warning(f"'{gu_name}'의 스케일된 마커 크기 조회 중 오류 ({scale_err}), 기본 크기 사용")

                folium.CircleMarker(
                    location=marker_location,
                    radius=radius,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.45,
                    tooltip=tooltip
                ).add_to(marker_group)
                
        logger.info("마커 추가 완료.")

    except Exception as e:
        logger.error(f"마커 생성 중 예상치 못한 오류: {e}")
        logger.debug(traceback.format_exc())

    # 5. 레이어 컨트롤 추가
    folium.LayerControl().add_to(folium_map)

    logger.info("Folium 지도 생성 완료.")
    return folium_map

def save_map_html(folium_map, output_dir: str, output_map_file: str):
    """생성된 Folium 지도를 HTML 파일로 저장합니다."""
    output_path = os.path.join(output_dir, output_map_file)
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"생성된 지도를 HTML 파일로 저장 중: {output_path}")
        folium_map.save(output_path)
        logger.info("지도 저장 완료.")
        return output_path
    except Exception as e:
        logger.error(f"지도 저장 중 오류 발생 ({output_path}): {str(e)}")
        logger.debug(traceback.format_exc())
        raise IOError(f"지도를 HTML 파일({output_path})로 저장 실패: {str(e)}")

def create_map(data_dir: str = 'app/up_data', 
                 output_dir: str = 'app/map_data', 
                 police_norm_filename: str = 'police_norm_in_seoul.csv', 
                 geo_json_filename: str = 'geo_simple.json',
                 cctv_filename: str = 'cctv_in_seoul.csv',
                 pop_filename: str = 'pop_in_seoul.csv',
                 output_map_filename: str = 'crime_map.html') -> str:
    """범죄 지도를 생성하고 저장된 파일 경로를 반환합니다."""
    try:
        logger.info("범죄 지도 생성 시작...")
        logger.info(f"데이터 디렉토리: {data_dir}, 출력 디렉토리: {output_dir}")

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        police_norm, state_geo, cctv_data, pop_data = load_required_data(
            data_dir, police_norm_filename, geo_json_filename, cctv_filename, pop_filename
        )
        
        folium_map = create_folium_map(police_norm, state_geo, cctv_data, pop_data)
        map_file_path = save_map_html(folium_map, output_dir, output_map_filename)

        logger.info(f"범죄 지도가 성공적으로 생성 및 저장되었습니다: {map_file_path}")
        return map_file_path
    except FileNotFoundError as e:
        logger.error(f"필수 파일 로드 실패: {e}")
        raise HTTPException(status_code=404, detail=f"필수 데이터 파일({e})을 찾을 수 없습니다.")
    except KeyError as e:
         logger.error(f"데이터 처리 중 필수 컬럼 부재: {e}")
         raise HTTPException(status_code=400, detail=f"데이터 처리 중 필요한 컬럼('{e}') 없음.")
    except ValueError as e:
         logger.error(f"데이터 처리 중 값 또는 형식 오류: {e}")
         raise HTTPException(status_code=400, detail=f"데이터 처리 오류: {e}")
    except IOError as e:
         logger.error(f"지도 파일 저장 실패: {e}")
         raise HTTPException(status_code=500, detail=f"지도 파일 저장 오류: {e}")
    except Exception as e:
        logger.error(f"지도 생성 중 예상치 못한 오류 발생: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"지도 생성 중 서버 오류: {type(e).__name__} - {str(e)}")




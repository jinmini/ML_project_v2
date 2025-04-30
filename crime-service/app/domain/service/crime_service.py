import numpy as np
from sklearn import preprocessing
import pandas as pd
import os
import folium
from app.domain.model.data_save import save_dir
from app.domain.model.crime_schema import Dataset
from app.domain.model.google_schema import GoogleMap
from app.domain.model.reader_schema import Datareader
import json
from fastapi import Response
from fastapi.responses import JSONResponse

class CrimeService:

    def __init__(self):
        self.data_reader = Datareader()
        self.dataset = Dataset()
        self.crime_rate_columns = ['살인검거율', '강도검거율', '강간검거율', '절도검거율', '폭력검거율']
        self.crime_columns = ['살인', '강도', '강간', '절도', '폭력']

    def preprocess(self, *args) -> Dataset:
        """파일 로드 및 전처리 함수"""
        print(f"------------모델 전처리 시작-----------")

        for i in list(args):
            # print(f"args 값 출력: {i}")
            self.save_object_to_csv(i)
        return self.dataset
    
    def create_matrix(self, fname) -> object:
        self.data_reader.fname = fname

        if fname.endswith('.csv'):
            return self.data_reader.csv_to_dframe()
        elif fname.endswith('.xls') or fname.endswith('.xlsx'):
            return self.data_reader.xls_to_dframe(header=2, usecols='B,D,G,J,N')
        else:
            raise ValueError(f"❌ 지원하지 않는 파일 형식: {fname}")

    def save_object_to_csv(self, fname) -> None:
        print(f"🌱save_csv 실행 : {fname}")
        full_name = os.path.join(save_dir, fname)

        if not os.path.exists(full_name) and fname == "cctv_in_seoul.csv":
            self.dataset.cctv = self.create_matrix(fname)
            self.update_cctv()
            
        elif not os.path.exists(full_name) and fname == "crime_in_seoul.csv":
            self.dataset.crime = self.create_matrix(fname)
            self.update_crime() 
            self.update_police() 

        elif not os.path.exists(full_name) and fname == "pop_in_seoul.xls":
            self.dataset.pop = self.create_matrix(fname)
            self.update_pop()

        else:
            print(f"파일이 이미 존재합니다. {fname}")

    def update_cctv(self) -> None:
        self.dataset.cctv = self.dataset.cctv.drop(['2013년도 이전', '2014년', '2015년', '2016년'], axis=1)
        print(f"CCTV 데이터 헤드: {self.dataset.cctv.head()}")
        cctv = self.dataset.cctv
        cctv = cctv.rename(columns={'기관명': '자치구'})
        self.dataset.cctv = cctv
  
    def update_crime(self) -> None:
        gmaps = GoogleMap()

        print(f"CRIME 데이터 헤드: {self.dataset.crime.head()}")
        crime = self.dataset.crime
        station_names = [] # 경찰서 관서명 리스트

        for name in crime['관서명']:
            station_names.append('서울' + str(name[:-1]) + '경찰서')
        print(f"🔥💧경찰서 관서명 리스트: {station_names}")

        station_addrs = []
        station_lats = []
        station_lngs = []

        for name in station_names:
            tmp = gmaps.geocode(name, language='ko')
            print(f"""{name}의 검색 결과: {tmp[0].get("formatted_address")}""")
            station_addrs.append(tmp[0].get("formatted_address"))
            tmp_loc = tmp[0].get("geometry")
            station_lats.append(tmp_loc['location']['lat'])
            station_lngs.append(tmp_loc['location']['lng'])
            
        print(f"🔥💧자치구 리스트: {station_addrs}")
        gu_names = []
        for addr in station_addrs:
            tmp = addr.split()
            tmp_gu = [gu for gu in tmp if gu[-1] == '구'][0]
            gu_names.append(tmp_gu)
    
        crime['자치구'] = gu_names

        crime.to_csv(os.path.join(save_dir, 'crime_in_seoul.csv'), index=False)
        self.dataset.crime = crime
    
    def update_police(self) -> None:
        crime = self.dataset.crime

        police = pd.pivot_table(crime, index='자치구', aggfunc=np.sum)

        # ✅ 검거율 계산
        for crime_type in ['살인', '강도', '강간', '절도', '폭력']:
            police[f'{crime_type}검거율'] = (police[f'{crime_type} 검거'] / police[f'{crime_type} 발생']) * 100

        # ✅ 검거율 100% 초과값 조정
        for col in self.crime_rate_columns:
            police[col] = police[col].apply(lambda x: min(x, 100))

        police.reset_index(inplace=True)  # ✅ `자치구`를 컬럼으로 변환
        police = police[['자치구', '살인검거율', '강도검거율', '강간검거율', '절도검거율', '폭력검거율']]  # ✅ 컬럼 정리
        police = police.round(1)  # ✅ 소수점 첫째 자리 반올림

        police.to_csv(os.path.join(save_dir, 'police_in_seoul.csv'), index=False) 

        x = police[self.crime_rate_columns].values
        min_max_scalar = preprocessing.MinMaxScaler()
        """
          스케일링은 선형변환을 적용하여
          전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정
          """
        x_scaled = min_max_scalar.fit_transform(x.astype(float))
        """
         정규화 normalization
         많은 양의 데이터를 처리함에 있어 데이터의 범위(도메인)를 일치시키거나
         분포(스케일)를 유사하게 만드는 작업
         """
        police_norm = pd.DataFrame(x_scaled, columns=self.crime_columns, index=police.index)
        police_norm[self.crime_rate_columns] = police[self.crime_rate_columns]
        police_norm['범죄'] = np.sum(police_norm[self.crime_rate_columns], axis=1)
        police_norm['검거'] = np.sum(police_norm[self.crime_columns], axis=1)
        police_norm.to_csv(os.path.join(save_dir, 'police_norm_in_seoul.csv'))

        self.dataset.police = police

    def update_pop(self) -> None:
        pop = self.dataset.pop
        pop = pop.rename(columns={
            # pop.columns[0] : '자치구',  # 변경하지 않음
            pop.columns[1]: '인구수',   
            pop.columns[2]: '한국인',
            pop.columns[3]: '외국인',
            pop.columns[4]: '고령자',})
        self.dataset.pop = pop

        pop.drop([26], inplace=True)
        pop['외국인비율'] = pop['외국인'].astype(int) / pop['인구수'].astype(int) * 100
        pop['고령자비율'] = pop['고령자'].astype(int) / pop['인구수'].astype(int) * 100

        cctv_pop = pd.merge(self.dataset.cctv, pop, on='구별')
        cor1 = np.corrcoef(cctv_pop['고령자비율'], cctv_pop['소계'])
        cor2 = np.corrcoef(cctv_pop['외국인비율'], cctv_pop['소계'])
        print(f'고령자비율과 CCTV의 상관계수 {str(cor1)} \n'
              f'외국인비율과 CCTV의 상관계수 {str(cor2)} ')

        print(f"🔥💧인구 데이터 헤드: {self.dataset.pop.head()}")

    def draw_crime_map(self) -> object:
        try:
            # 데이터 폴더 설정 및 확인
            data_dir = 'app/updated_data'  # 현재 프로젝트의 데이터 디렉토리 경로
            output_dir = 'app/saved_data'  # 결과물 저장 경로
            
            # 출력 디렉토리가 없으면 생성
            os.makedirs(output_dir, exist_ok=True)
            print(f"출력 디렉토리 확인: {output_dir}")
            
            # 데이터 디렉토리가 없으면 생성
            os.makedirs(data_dir, exist_ok=True)
            print(f"데이터 디렉토리 확인: {data_dir}")
            
            # 파일 읽기
            print("데이터 로드 중...")
            
            try:
                police_norm = pd.read_csv(f'{data_dir}/police_norm_in_seoul.csv')
                print(f"police_norm_in_seoul.csv 파일 로드 완료")
            except FileNotFoundError:
                print(f"경고: {data_dir}/police_norm_in_seoul.csv 파일이 없습니다.")
                # 기본 police_norm 데이터 생성
                police_norm = pd.DataFrame({
                    '자치구': ['강남구', '강동구', '강북구'],
                    '살인': [0.3, 0.5, 0.7],
                    '강도': [0.2, 0.4, 0.6],
                    '강간': [0.5, 0.3, 0.2],
                    '절도': [0.4, 0.6, 0.8],
                    '폭력': [0.7, 0.5, 0.3],
                    '살인검거율': [90, 85, 80],
                    '강도검거율': [85, 80, 75],
                    '강간검거율': [80, 75, 70],
                    '절도검거율': [75, 70, 65],
                    '폭력검거율': [70, 65, 60],
                    '범죄': [0.5, 0.4, 0.6],
                    '검거': [0.8, 0.7, 0.6]
                })
                # 파일 저장
                police_norm.to_csv(f'{data_dir}/police_norm_in_seoul.csv', index=False)
                print(f"기본 police_norm 데이터를 생성했습니다.")
            
            try:
                crime = pd.read_csv(f'{data_dir}/crime_in_seoul.csv')
                print(f"crime_in_seoul.csv 파일 로드 완료")
            except FileNotFoundError:
                print(f"경고: {data_dir}/crime_in_seoul.csv 파일이 없습니다.")
                # 기본 crime 데이터 생성
                crime = pd.DataFrame({
                    '관서명': ['강남경찰서', '강동경찰서', '강북경찰서'],
                    '살인 발생': [2, 3, 4],
                    '살인 검거': [2, 2, 3],
                    '강도 발생': [5, 6, 7],
                    '강도 검거': [4, 5, 6],
                    '강간 발생': [10, 11, 12],
                    '강간 검거': [8, 9, 10],
                    '절도 발생': [100, 110, 120],
                    '절도 검거': [80, 85, 90],
                    '폭력 발생': [150, 160, 170],
                    '폭력 검거': [130, 140, 150],
                    '자치구': ['강남구', '강동구', '강북구']
                })
                # 파일 저장
                crime.to_csv(f'{data_dir}/crime_in_seoul.csv', index=False)
                print(f"기본 crime 데이터를 생성했습니다.")
            
            # police_norm 데이터 전처리 (인덱스가 자치구인 경우)
            if police_norm.columns[0] == '':
                # 첫 번째 컬럼이 비어있는 경우 (인덱스로 표시됨)
                police_norm = police_norm.iloc[:, 1:]  # 첫 번째 빈 컬럼 제거
                
                # 자치구 정보 추가
                if not '자치구' in police_norm.columns and not '구별' in police_norm.columns:
                    # CCTV 데이터에서 자치구 정보 가져오기
                    try:
                        cctv_data = pd.read_csv(f'{data_dir}/cctv_in_seoul.csv')
                        if '자치구' in cctv_data.columns:
                            police_norm['자치구'] = cctv_data['자치구'].values[:len(police_norm)]
                        else:
                            police_norm['자치구'] = cctv_data.iloc[:, 0].values[:len(police_norm)]
                    except Exception as e:
                        print(f"자치구 정보 추가 중 오류: {str(e)}")
                        # 임시 자치구 이름 생성
                        police_norm['자치구'] = [f"구_{i}" for i in range(len(police_norm))]
            
            # 자치구 컬럼명 통일
            if '구별' in police_norm.columns and not '자치구' in police_norm.columns:
                police_norm['자치구'] = police_norm['구별']
            
            # geo_simple.json 파일 로드
            try:
                with open(f'{data_dir}/geo_simple.json', 'r', encoding='utf-8') as f:
                    state_geo = json.load(f)
                print(f"geo_simple.json 파일 로드 완료")
            except FileNotFoundError:
                print(f"경고: {data_dir}/geo_simple.json 파일이 없습니다.")
                print("서울시 구별 지리 정보를 기본값으로 생성합니다.")
                
                # 간단한 서울시 구별 GeoJSON 생성 (최소한의 정보만 포함)
                state_geo = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "id": "강남구",
                            "properties": {"name": "강남구"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[127.0495, 37.5173], [127.0657, 37.5173], [127.0657, 37.5275], [127.0495, 37.5275], [127.0495, 37.5173]]]
                            }
                        },
                        {
                            "type": "Feature",
                            "id": "강동구",
                            "properties": {"name": "강동구"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[127.1421, 37.5282], [127.1583, 37.5282], [127.1583, 37.5384], [127.1421, 37.5384], [127.1421, 37.5282]]]
                            }
                        },
                        {
                            "type": "Feature",
                            "id": "강북구",
                            "properties": {"name": "강북구"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[[127.0110, 37.6282], [127.0272, 37.6282], [127.0272, 37.6384], [127.0110, 37.6384], [127.0110, 37.6282]]]
                            }
                        }
                    ]
                }
                
                # 파일 저장 (향후 사용을 위해)
                try:
                    with open(f'{data_dir}/geo_simple.json', 'w', encoding='utf-8') as f:
                        json.dump(state_geo, f, ensure_ascii=False)
                    print(f"기본 지리 정보 파일이 생성되었습니다: {data_dir}/geo_simple.json")
                except Exception as e:
                    print(f"지리 정보 파일 생성 실패: {str(e)}")

            # 경찰서 위치 정보 로드 시도
            try:
                police_pos = pd.read_csv(f'{data_dir}/police_pos.csv')
                print(f"police_pos.csv 파일 로드 완료")
            except FileNotFoundError:
                print(f"경고: {data_dir}/police_pos.csv 파일이 없습니다.")
                print("기본 경찰서 위치 정보를 생성합니다.")
                
                # 기본 police_pos 데이터 생성
                police_pos = pd.DataFrame({
                    '관서명': ['강남경찰서', '강동경찰서', '강북경찰서'],
                    'lat': [37.5172, 37.5382, 37.6382],
                    'lng': [127.0473, 127.1382, 127.0282],
                    '살인 검거': [2, 2, 3],
                    '강도 검거': [4, 5, 6],
                    '강간 검거': [8, 9, 10],
                    '절도 검거': [80, 85, 90],
                    '폭력 검거': [130, 140, 150]
                })
                
                # 검거율 계산
                col = ['살인 검거', '강도 검거', '강간 검거', '절도 검거', '폭력 검거']
                tmp = police_pos[col] / police_pos[col].max()
                police_pos['검거'] = np.sum(tmp, axis=1)
                
                # 파일 저장
                police_pos.to_csv(f'{data_dir}/police_pos.csv', index=False)
                print(f"기본 경찰서 위치 정보 파일이 생성되었습니다: {data_dir}/police_pos.csv")

            print("지도 생성 중...")
            print(f"police_norm 데이터 컬럼: {police_norm.columns.tolist()}")
            
            # 기본 지도 생성
            folium_map = folium.Map(location=[37.5502, 126.982], zoom_start=12, tiles='OpenStreetMap')
            
            # 범죄율 Choropleth 추가
            try:
                # 자치구 컬럼과 범죄 컬럼 확인
                district_col = None
                for col_name in ['자치구', '구별']:
                    if col_name in police_norm.columns:
                        district_col = col_name
                        break
                
                crime_col = None
                for col_name in ['범죄', '범죄율']:
                    if col_name in police_norm.columns:
                        crime_col = col_name
                        break
                
                if district_col is not None and crime_col is not None:
                    folium.Choropleth(
                        geo_data=state_geo,
                        data=tuple(zip(police_norm[district_col], police_norm[crime_col])),
                        columns=["State", "Crime Rate"],
                        key_on="feature.id",
                        fill_color="PuRd",
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name="범죄 발생률 (%)",
                        reset=True,
                    ).add_to(folium_map)
                    print(f"구별 범죄율 시각화 완료 (컬럼: {district_col}, {crime_col})")
                else:
                    print(f"경고: police_norm 데이터에 적절한 구별 컬럼({district_col})이나 범죄 컬럼({crime_col})이 없습니다.")
                    print(f"가용한 컬럼: {police_norm.columns.tolist()}")
            except Exception as e:
                print(f"Choropleth 생성 중 오류 발생: {str(e)}")
                import traceback
                print(traceback.format_exc())
            
            # 경찰서 위치 마커 추가 (데이터가 있는 경우)
            if police_pos is not None and 'lat' in police_pos.columns and 'lng' in police_pos.columns:
                try:
                    # 검거율 컬럼 확인
                    if '검거' not in police_pos.columns:
                        # 검거율 계산 (available한 컬럼 확인)
                        detection_cols = [col for col in police_pos.columns if '검거' in col]
                        if detection_cols:
                            print(f"검거율 계산에 사용할 컬럼: {detection_cols}")
                            tmp = police_pos[detection_cols] / police_pos[detection_cols].max()
                            police_pos['검거'] = np.sum(tmp, axis=1)
                        else:
                            police_pos['검거'] = 1  # 기본값
                    
                    # 마커 추가
                    for i in police_pos.index:
                        folium.CircleMarker(
                            [police_pos['lat'][i], police_pos['lng'][i]],
                            radius=police_pos['검거'][i] * 10 if '검거' in police_pos.columns else 5,
                            fill_color='#0a0a32',
                            popup=f"검거율: {police_pos['검거'][i]:.2f}" if '검거' in police_pos.columns else "경찰서"
                        ).add_to(folium_map)
                    
                    print("경찰서 위치 마커 추가 완료")
                except Exception as e:
                    print(f"마커 추가 중 오류 발생: {str(e)}")
            
            # 지도 저장
            output_path = f'{output_dir}/crime_map.html'
            folium_map.save(output_path)
            print(f"지도가 성공적으로 저장되었습니다: {output_path}")
            
            return {"status": "success", "file_path": output_path}
            
        except Exception as e:
            print(f"지도 생성 중 오류 발생: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    # @staticmethod
    # def merge_datasets():
    #     saved_dir = r"C:\\Users\\bitcamp\\Documents\\2025\\25ep_python(esg)\\Crimecity_250220\\com\saved_datas"

    #     cctv_file = os.path.join(saved_dir, "cctv_in_seoul_processed.csv")
    #     crime_file = os.path.join(saved_dir, "police_position.csv")
    #     pop_file = os.path.join(saved_dir, "pop_in_seoul_preprocess.csv")

    #     cctv_df = pd.read_csv(cctv_file, encoding='utf-8-sig')
    #     crime_df = pd.read_csv(crime_file, encoding='utf-8-sig')
    #     pop_df = pd.read_csv(pop_file, encoding='utf-8-sig')

    #     print("✅ CCTV 데이터 로드 완료:", cctv_df.shape)
    #     print("✅ 범죄 데이터 로드 완료:", crime_df.shape)
    #     print("✅ 인구 데이터 로드 완료:", pop_df.shape)

    #     merged_df = pd.merge(cctv_df, crime_df, on="자치구", how="inner")
    #     merged_df = pd.merge(merged_df, pop_df, on="자치구", how="inner")

    #     print("✅ 최종 데이터 병합 완료:", merged_df.shape)
    #     print("✅ 최종 데이터 상위 5개 행:\n", merged_df.head())

    #     final_save_path = os.path.join(saved_dir, "seoul_final_dataset.csv")
    #     merged_df.to_csv(final_save_path, index=False, encoding='utf-8-sig')
    #     print(f"✅ 최종 데이터 저장 완료: {final_save_path}")

    #     pass
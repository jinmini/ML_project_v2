import pandas as pd
import numpy as np
import os
import traceback
import logging 

logger = logging.getLogger(__name__)

def get_interpretation_text(corr, var1, var2):
    """상관계수 해석 텍스트를 반환하는 함수"""
    if pd.isna(corr): # NaN 값 처리
         return f"{var1}과 {var2} 사이의 상관계수를 계산할 수 없습니다 (데이터 부족 또는 오류)."

    if abs(corr) < 0.1:
        strength = "거의 없음 (매우 약함)"
    elif abs(corr) < 0.3:
        strength = "약함"
    elif abs(corr) < 0.5:
        strength = "중간"
    elif abs(corr) < 0.7:
        strength = "강함"
    else:
        strength = "매우 강함"

    direction = "양의" if corr > 0 else "음의" if corr < 0 else "없는"

    if direction == "없는":
         return f"{var1}과 {var2} 사이에는 통계적으로 유의미한 선형 상관관계가 없습니다."
    else:
         tendency = '증가' if corr > 0 else '감소'
         return f"{var1}과 {var2} 사이에는 {strength} 강도의 {direction} 선형 상관관계가 있습니다. 즉, {var1}이(가) 증가할 때 {var2}도 {tendency}하는 경향을 보입니다."

def analyze_correlation(cctv_data, pop_data):
    """CCTV와 인구 데이터의 상관관계를 분석하는 함수"""
    try:
        logger.info("인구 통계 상관관계 분석 시작...") # 로깅으로 변경
        cctv_data = cctv_data.copy() # 원본 보호
        pop_data = pop_data.copy()

        if '자치구' in cctv_data.columns and '자치구' in pop_data.columns:
            merge_col = '자치구'
        elif '구별' in cctv_data.columns and '구별' in pop_data.columns:
            merge_col = '구별'
            # '구별'을 '자치구'로 통일
            cctv_data = cctv_data.rename(columns={'구별': '자치구'})
            pop_data = pop_data.rename(columns={'구별': '자치구'})
        else:
            # 첫 번째 컬럼을 기준으로 통일
            first_col_cctv = cctv_data.columns[0]
            first_col_pop = pop_data.columns[0]
            logger.warning(f"컬럼명이 일치하지 않아 첫 번째 컬럼({first_col_cctv}, {first_col_pop})을 '자치구'로 통일합니다.") # 로깅으로 변경
            cctv_data = cctv_data.rename(columns={first_col_cctv: '자치구'})
            pop_data = pop_data.rename(columns={first_col_pop: '자치구'})
            merge_col = '자치구'

        logger.info(f"데이터 병합에 사용할 컬럼: {merge_col}") 

        # 인구 데이터 컬럼명 변경 및 전처리
        try:
             pop_data = pop_data.rename(columns={
                 pop_data.columns[1]: '인구수',
                 pop_data.columns[2]: '한국인',
                 pop_data.columns[3]: '외국인',
                 pop_data.columns[4]: '고령자',
             })
        except IndexError:
             logger.error("인구 데이터에 필요한 컬럼(인구수, 한국인, 외국인, 고령자)이 부족합니다.") 
             raise ValueError("인구 데이터에 필요한 컬럼(인구수, 한국인, 외국인, 고령자)이 부족합니다.")


        if '합계' in pop_data['자치구'].values:
             pop_data = pop_data[pop_data['자치구'] != '합계'].copy()
             logger.info("인구 데이터에서 '합계' 행 제거됨") 

        # 숫자형 변환 및 결측치 처리
        for col in ['인구수', '한국인', '외국인', '고령자']:
             pop_data[col] = pd.to_numeric(pop_data[col], errors='coerce')
        pop_data = pop_data.fillna(0) # NaN을 0으로 채움 (비율 계산 전)


        pop_data['외국인비율'] = np.where(pop_data['인구수'] > 0, (pop_data['외국인'] / pop_data['인구수']) * 100, 0)
        pop_data['고령자비율'] = np.where(pop_data['인구수'] > 0, (pop_data['고령자'] / pop_data['인구수']) * 100, 0)

        # 병합 (CCTV는 자치구와 개수 컬럼만 사용)
        cctv_merge_col = cctv_data.columns[1] # CCTV 개수 컬럼 (원본 기준 두 번째)
        cctv_pop = pd.merge(cctv_data[['자치구', cctv_merge_col]], pop_data, on=merge_col, how='inner')
        if cctv_pop.empty:
            logger.error("CCTV 데이터와 인구 데이터 병합 결과가 비어 있습니다.") # 로깅
            return {'error': 'CCTV와 인구 데이터 병합 실패', 'details': '자치구 이름 불일치 가능성'}
        logger.info(f"병합된 인구 데이터 형태: {cctv_pop.shape}, 컬럼: {cctv_pop.columns.tolist()}") # 로깅

        # CCTV 개수 컬럼명 통일 ('CCTV개수') 및 타입 확인
        if cctv_merge_col in cctv_pop.columns:
             cctv_col = 'CCTV개수'
             cctv_pop = cctv_pop.rename(columns={cctv_merge_col: cctv_col})
             logger.info(f"CCTV 컬럼명을 '{cctv_col}'(으)로 변경") # 로깅

             if not pd.api.types.is_numeric_dtype(cctv_pop[cctv_col]):
                 logger.warning(f"CCTV 컬럼 '{cctv_col}'이 숫자형이 아님. 형변환 시도.") # 로깅
                 cctv_pop[cctv_col] = pd.to_numeric(cctv_pop[cctv_col], errors='coerce')
                 if cctv_pop[cctv_col].isnull().any():
                      logger.error(f"CCTV 컬럼 '{cctv_col}' 숫자형 변환 실패. NaN 포함 행 제거.") # 로깅
                      cctv_pop = cctv_pop.dropna(subset=[cctv_col])
                      if cctv_pop.empty: raise ValueError("CCTV 개수 변환 실패 후 데이터 없음")
        else:
             logger.error(f"병합된 데이터프레임에서 원본 CCTV 개수 컬럼({cctv_merge_col})을 찾을 수 없습니다.") # 로깅
             raise ValueError(f"병합된 데이터프레임에서 원본 CCTV 개수 컬럼({cctv_merge_col})을 찾을 수 없습니다.")

        # 상관계수 계산 전 NaN 및 데이터 수 확인
        cols_for_corr = ['고령자비율', '외국인비율', cctv_col]
        if cctv_pop[cols_for_corr].isnull().any().any():
             logger.warning("상관계수 계산 대상 컬럼에 NaN 값 포함. 해당 행 제거.") # 로깅
             cctv_pop = cctv_pop.dropna(subset=cols_for_corr)
             logger.info(f"NaN 값 제거 후 데이터 형태: {cctv_pop.shape}") # 로깅

        if len(cctv_pop) < 2:
             logger.error("상관계수 계산 데이터 부족 (최소 2개 필요).") # 로깅
             return {
                 'error': '데이터 부족으로 인구 통계 상관관계 분석 실패',
                 'elderly_correlation': None,
                 'foreigner_correlation': None,
                 'cctv_correlations': [],
                 'districts': []
             }

        # 특정 두 변수 상관계수
        cor1 = np.corrcoef(cctv_pop['고령자비율'], cctv_pop[cctv_col])
        cor2 = np.corrcoef(cctv_pop['외국인비율'], cctv_pop[cctv_col])
        cor1_val = cor1[0, 1] if cor1.shape == (2, 2) else np.nan
        cor2_val = cor2[0, 1] if cor2.shape == (2, 2) else np.nan

        logger.info(f"고령자비율-CCTV 상관계수: {cor1_val:.4f}") # 로깅
        logger.info(f"외국인비율-CCTV 상관계수: {cor2_val:.4f}") # 로깅

        # 해석 (get_interpretation_text 함수 직접 호출)
        elderly_interpretation = get_interpretation_text(cor1_val, "고령자비율", "CCTV")
        foreigner_interpretation = get_interpretation_text(cor2_val, "외국인비율", "CCTV")

        logger.info(f"해석 (고령자): {elderly_interpretation}") # 로깅
        logger.info(f"해석 (외국인): {foreigner_interpretation}") # 로깅

        # CCTV와 다른 숫자형 변수들 간 상관관계
        logger.info("===== CCTV와 다른 변수들 간 상관관계 분석 =====") # 로깅
        numeric_columns = cctv_pop.select_dtypes(include=np.number).columns
        cols_to_analyze = [col for col in numeric_columns
                           if col not in [cctv_col, merge_col, '한국인', '외국인', '고령자']]

        cctv_correlations = []

        for col in cols_to_analyze:
            if len(cctv_pop) < 2: continue

            try:
                 corr_value = np.corrcoef(cctv_pop[col], cctv_pop[cctv_col])[0, 1]
                 interpretation = get_interpretation_text(corr_value, col, "CCTV")

                 logger.debug(f"{col} - {cctv_col}: {corr_value:.4f}") 
                 logger.debug(f"  해석: {interpretation}") 

                 cctv_correlations.append({
                     'variable': str(col),
                     'correlation': float(f"{corr_value:.4f}"),
                     'interpretation': interpretation
                 })
            except Exception as corr_err:
                 logger.warning(f"오류: 컬럼 '{col}'과 CCTV 간 상관계수 계산 중 - {corr_err}") 


        cctv_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        # 결과 반환
        result = {
            'elderly_correlation': {
                'value': float(f'{cor1_val:.4f}') if pd.notna(cor1_val) else None,
                'interpretation': elderly_interpretation
            },
            'foreigner_correlation': {
                'value': float(f'{cor2_val:.4f}') if pd.notna(cor2_val) else None,
                'interpretation': foreigner_interpretation
            },
            'cctv_correlations': cctv_correlations,
            'districts': [
                {
                    'district': row[merge_col],
                    'cctv_count': int(row[cctv_col]) if pd.notna(row[cctv_col]) else None,
                    'population': int(row['인구수']) if pd.notna(row['인구수']) else None,
                    'elderly_ratio': float(f'{row["고령자비율"]:.2f}') if pd.notna(row["고령자비율"]) else None,
                    'foreigner_ratio': float(f'{row["외국인비율"]:.2f}') if pd.notna(row["외국인비율"]) else None
                }
                for _, row in cctv_pop.iterrows()
            ]
        }
        return result

    except KeyError as e:
         logger.error(f"인구 통계 상관계수 계산 중 오류 발생: 필요한 컬럼({e})을 찾을 수 없습니다.") # 로깅
         # print(traceback.format_exc())
         return {'error': f"KeyError: 필요한 컬럼({e})을 찾을 수 없습니다.", 'details': traceback.format_exc()}
    except ValueError as e:
         logger.error(f"인구 통계 상관계수 계산 중 값 오류 발생: {str(e)}") # 로깅
         # print(traceback.format_exc())
         return {'error': f"ValueError: {str(e)}", 'details': traceback.format_exc()}
    except Exception as e:
        logger.error(f"인구 통계 상관계수 계산 중 예상치 못한 오류 발생: {str(e)}") # 로깅
        # print(traceback.format_exc())
        return {'error': f"Exception: {str(e)}", 'details': traceback.format_exc()}


def analyze_crime_correlation(cctv_data, crime_data, police_norm_data):
    """CCTV와 범죄 데이터의 상관관계를 분석하는 함수"""
    try:
        logger.info("===== CCTV와 범죄 데이터 상관관계 분석 시작 =====") # 로깅

        cctv_data = cctv_data.copy()
        crime_data = crime_data.copy()
        police_norm_data = police_norm_data.copy()

        # --- CCTV 데이터 전처리 ---
        if '자치구' not in cctv_data.columns:
            if '구별' in cctv_data.columns:
                cctv_data = cctv_data.rename(columns={'구별': '자치구'})
                logger.info("CCTV 데이터: '구별' -> '자치구' 변경") # 로깅
            elif len(cctv_data.columns) > 0:
                first_col = cctv_data.columns[0]
                cctv_data = cctv_data.rename(columns={first_col: '자치구'})
                logger.warning(f"CCTV 데이터: 첫 컬럼 '{first_col}' -> '자치구' 변경") # 로깅
            else: raise ValueError("CCTV 데이터에 '자치구' 식별 컬럼 없음")

        cctv_col = '소계' if '소계' in cctv_data.columns else cctv_data.columns[1]
        if not pd.api.types.is_numeric_dtype(cctv_data[cctv_col]):
            logger.warning(f"경고: CCTV 컬럼 '{cctv_col}' 숫자형 변환 시도") # 로깅
            cctv_data[cctv_col] = pd.to_numeric(cctv_data[cctv_col], errors='coerce')
            if cctv_data[cctv_col].isnull().any():
                 logger.error(f"오류: CCTV 컬럼 '{cctv_col}' 숫자형 변환 실패. NaN 포함 행 제거.") # 로깅
                 cctv_data = cctv_data.dropna(subset=[cctv_col])
                 if cctv_data.empty: raise ValueError("CCTV 개수 변환 실패 후 데이터 없음")
        logger.info(f"CCTV 데이터: '자치구' 컬럼 = '자치구', 개수 컬럼 = '{cctv_col}' 사용") # 로깅


        # --- 범죄 데이터 전처리 ---
        if '자치구' not in crime_data.columns:
             if '구별' in crime_data.columns:
                  crime_data = crime_data.rename(columns={'구별': '자치구'})
                  logger.info("범죄 데이터: '구별' -> '자치구' 변경") # 로깅
             else:
                  logger.error("범죄 데이터에 '자치구' 컬럼이 필요합니다 (전처리 필요).") # 로깅
                  raise ValueError("범죄 데이터에 '자치구' 컬럼이 필요합니다 (전처리 필요).")
        crime_occur_cols = [col for col in crime_data.columns if col.endswith(' 발생')]
        if not crime_occur_cols:
            logger.error("범죄 데이터에 ' 발생' 컬럼 없음") # 로깅
            raise ValueError("범죄 데이터에 ' 발생' 컬럼 없음")
        for col in crime_occur_cols:
            crime_data[col] = pd.to_numeric(crime_data[col], errors='coerce').fillna(0)
        crime_by_district = crime_data.groupby('자치구')[crime_occur_cols].sum().reset_index()
        logger.info(f"범죄 데이터: '자치구' 컬럼 및 발생 건수({len(crime_occur_cols)}개) 확인 및 집계 완료") # 로깅


        # --- 경찰서 정규화 데이터 전처리 ---
        if '자치구' not in police_norm_data.columns:
             if '구별' in police_norm_data.columns:
                  police_norm_data = police_norm_data.rename(columns={'구별': '자치구'})
                  logger.info("경찰서 데이터: '구별' -> '자치구' 변경") # 로깅
             elif 'Unnamed: 0' in police_norm_data.columns and pd.api.types.is_string_dtype(police_norm_data['Unnamed: 0']):
                  police_norm_data = police_norm_data.rename(columns={'Unnamed: 0': '자치구'})
                  logger.info("경찰서 데이터: 'Unnamed: 0' -> '자치구' 사용") # 로깅
             else:
                 logger.error("경찰서 정규화 데이터에 '자치구' 식별 컬럼 없음") # 로깅
                 raise ValueError("경찰서 정규화 데이터에 '자치구' 식별 컬럼 없음")

        # 검거율 컬럼 동적 식별 및 처리
        crime_rate_cols = [col for col in police_norm_data.columns if col.endswith('검거율')]
        logger.info(f"자동 식별된 검거율 컬럼: {crime_rate_cols}") # 로깅
        police_cols_to_check = crime_rate_cols + ['범죄', '검거'] # '범죄', '검거' 컬럼은 여전히 필요

        for col in police_cols_to_check:
            if col in police_norm_data.columns:
                police_norm_data[col] = pd.to_numeric(police_norm_data[col], errors='coerce').fillna(0)
        logger.info("경찰서 데이터: '자치구' 컬럼 및 필요 컬럼 확인/처리 완료") # 로깅


        # --- 데이터 병합 ---
        merged_data = pd.merge(cctv_data[['자치구', cctv_col]], crime_by_district, on='자치구', how='inner')
        if merged_data.empty:
            logger.error("CCTV와 범죄 데이터 병합 결과 없음") # 로깅
            raise ValueError("CCTV와 범죄 데이터 병합 결과 없음")
        logger.info(f"CCTV+범죄 병합 후: {len(merged_data)} 행") # 로깅

        police_cols_to_merge = ['자치구'] + [col for col in police_cols_to_check if col in police_norm_data.columns]
        if police_norm_data['자치구'].duplicated().any():
            logger.warning("경고: 경찰서 정규화 데이터에 중복된 자치구가 있습니다. 첫 번째 값만 사용합니다.") # 로깅
            police_norm_data = police_norm_data.drop_duplicates(subset=['자치구'], keep='first')

        merged_data = pd.merge(merged_data, police_norm_data[police_cols_to_merge], on='자치구', how='inner')
        if merged_data.empty:
            logger.error("최종 데이터 병합 결과 없음") # 로깅
            raise ValueError("최종 데이터 병합 결과 없음")
        logger.info(f"최종 병합 후: {len(merged_data)} 행, 컬럼: {merged_data.columns.tolist()}") # 로깅


        # --- 상관관계 분석 ---
        crime_vars_to_analyze = crime_occur_cols + [col for col in police_cols_to_check if col in merged_data.columns]
        logger.info("===== CCTV와 범죄/검거 관련 변수 상관관계 =====") # 로깅
        crime_correlations = []

        if len(merged_data) < 2:
             logger.warning("경고: 최종 병합 데이터가 2개 미만이라 상관관계 계산 불가") # 로깅
        else:
             for col in crime_vars_to_analyze:
                 try:
                     series1 = merged_data[col].dropna()
                     series2 = merged_data[cctv_col].dropna()
                     common_index = series1.index.intersection(series2.index)
                     if len(common_index) < 2:
                          logger.warning(f"경고: 컬럼 '{col}'과 CCTV 간 공통 유효 데이터 부족 (<2), 상관관계 계산 건너뜀") # 로깅
                          continue

                     corr_value = np.corrcoef(series1.loc[common_index], series2.loc[common_index])[0, 1]
                     interpretation = get_interpretation_text(corr_value, col, "CCTV")

                     logger.debug(f"{col} - CCTV: {corr_value:.4f}") # 디버그 레벨 로깅
                     logger.debug(f"  해석: {interpretation}") # 디버그 레벨 로깅

                     crime_correlations.append({
                         'variable': str(col),
                         'correlation': float(f"{corr_value:.4f}") if pd.notna(corr_value) else None,
                         'interpretation': interpretation
                     })
                 except Exception as corr_err:
                     logger.warning(f"오류: 컬럼 '{col}'과 CCTV 간 상관계수 계산 중 - {corr_err}") # 로깅

        crime_correlations.sort(key=lambda x: abs(x['correlation'] if x['correlation'] is not None else 0), reverse=True)
        logger.info("===== CCTV와 범죄 데이터 상관관계 분석 완료 =====") # 로깅


        # --- 결과 반환 ---
        result = {
            'crime_correlations': crime_correlations,
            'districts': [
                {
                    'district': row['자치구'],
                    'cctv_count': int(row[cctv_col]) if pd.notna(row[cctv_col]) else None,
                    **{col: float(row[col]) if pd.notna(row[col]) else None
                       for col in crime_vars_to_analyze if col in merged_data.columns and col in row}
                }
                for _, row in merged_data.iterrows()
            ]
        }
        return result

    except KeyError as e:
         logger.error(f"범죄 상관계수 계산 중 오류 발생: 필요한 컬럼({e})을 찾을 수 없습니다.") # 로깅
         # print(traceback.format_exc())
         return {'error': f"KeyError: 필요한 컬럼({e})을 찾을 수 없습니다.", 'details': traceback.format_exc()}
    except ValueError as e:
         logger.error(f"범죄 상관계수 계산 중 값 오류 발생: {str(e)}") # 로깅
         # print(traceback.format_exc())
         return {'error': f"ValueError: {str(e)}", 'details': traceback.format_exc()}
    except Exception as e:
        logger.error(f"범죄 상관계수 계산 중 예상치 못한 오류 발생: {str(e)}") # 로깅
        # print(traceback.format_exc())
        return {'error': f"Exception: {str(e)}", 'details': traceback.format_exc()}


def load_and_analyze(data_dir='app/up_data'):
    """데이터를 로드하고 상관관계를 분석하는 함수"""
    try:
        logger.info("===== 상관계수 분석 로드 및 실행 시작 =====") # 로깅

        cctv_file = os.path.join(data_dir, 'cctv_in_seoul.csv')
        pop_file = os.path.join(data_dir, 'pop_in_seoul.csv')
        crime_file = os.path.join(data_dir, 'crime_in_seoul_updated.csv')
        police_norm_file = os.path.join(data_dir, 'police_norm_in_seoul.csv')

        required_files = { "CCTV": cctv_file, "Population": pop_file, "Crime": crime_file, "Police Norm": police_norm_file }
        for name, path in required_files.items():
             if not os.path.exists(path):
                  logger.error(f"필수 데이터 파일 '{name}' 없음: {path}") # 로깅
                  raise FileNotFoundError(f"필수 데이터 파일 '{name}' 없음: {path}")

        # 데이터 로드
        logger.info("데이터 로드 중...") 
        cctv_data = pd.read_csv(cctv_file)
        pop_data = pd.read_csv(pop_file)
        crime_data = pd.read_csv(crime_file)
        police_norm_data = pd.read_csv(police_norm_file)
        logger.info("데이터 로드 완료.") 
        logger.info(f"  CCTV: {cctv_data.shape}, Pop: {pop_data.shape}, Crime: {crime_data.shape}, Police: {police_norm_data.shape}") # 로깅

        # 분석 수행
        demographic_results = {}
        crime_correlation_results = {}
        error_occurred = False
        error_messages = []

        try:
             logger.info("--- 인구 통계 데이터 상관관계 분석 ---") 
             demographic_results = analyze_correlation(cctv_data, pop_data) # 함수 직접 호출
             if isinstance(demographic_results, dict) and 'error' in demographic_results:
                 error_occurred = True
                 error_messages.append(f"인구 통계 분석 오류: {demographic_results['error']}")
                 logger.warning(f"인구 통계 분석 중 오류 발생: {demographic_results['error']}") 
        except Exception as demo_e:
             logger.error(f"심각한 오류: 인구 통계 분석 중 예외 발생 - {demo_e}") 
             demographic_results = {"error": f"심각한 예외: {str(demo_e)}", 'details': traceback.format_exc()}
             error_occurred = True
             error_messages.append(f"인구 통계 분석 실패 (예외)")

        try:
             logger.info("--- 범죄 데이터 상관관계 분석 ---") # 로깅
             crime_correlation_results = analyze_crime_correlation(cctv_data, crime_data, police_norm_data) # 함수 직접 호출
             if isinstance(crime_correlation_results, dict) and 'error' in crime_correlation_results:
                 error_occurred = True
                 error_messages.append(f"범죄 데이터 분석 오류: {crime_correlation_results['error']}")
                 logger.warning(f"범죄 데이터 분석 중 오류 발생: {crime_correlation_results['error']}") # 로깅
        except Exception as crime_e:
             logger.error(f"심각한 오류: 범죄 데이터 분석 중 예외 발생 - {crime_e}") # 로깅
             crime_correlation_results = {"error": f"심각한 예외: {str(crime_e)}", 'details': traceback.format_exc()}
             error_occurred = True
             error_messages.append(f"범죄 데이터 분석 실패 (예외)")


        # 결과 통합
        results = {
            'demographic_analysis': demographic_results,
            'crime_analysis': crime_correlation_results
        }

        if error_occurred:
             results['status'] = 'Partial Failure'
             results['message'] = f"일부 상관관계 분석 중 오류 발생: {'; '.join(error_messages)}"
             logger.warning(f"분석 결과: 부분 실패 - {results['message']}") # 로깅
        else:
             final_status = 'Success'
             final_message = '모든 상관관계 분석이 성공적으로 완료되었습니다.'
             # 세부 오류는 이미 로깅되었으므로 최종 상태만 기록
             if isinstance(demographic_results, dict) and 'error' in demographic_results: final_status = 'Partial Failure'
             if isinstance(crime_correlation_results, dict) and 'error' in crime_correlation_results: final_status = 'Partial Failure'

             results['status'] = final_status
             results['message'] = final_message if final_status == 'Success' else f"분석 중 일부 오류 발생 (세부사항은 로그 확인)"
             if final_status == 'Success':
                  logger.info("분석 결과: 성공") # 로깅
             else:
                  logger.warning(f"분석 결과: 부분 실패 - {results['message']}") # 로깅


        logger.info("===== 상관계수 분석 로드 및 실행 완료 =====") # 로깅
        return results

    except FileNotFoundError as e:
         logger.error(f"상관관계 분석 중 오류 발생: {str(e)}") # 로깅
         return { 'status': 'Failure', 'error': str(e), 'message': '필수 데이터 파일을 찾을 수 없습니다.' }
    except Exception as e:
        logger.error(f"상관관계 분석 로드 및 실행 중 예상치 못한 오류 발생: {str(e)}") # 로깅
        # print(traceback.format_exc()) # traceback은 필요한 경우에만 로깅
        logger.debug(traceback.format_exc()) # 디버그 레벨로 스택 트레이스 로깅
        return { 'status': 'Failure', 'error': str(e), 'traceback': traceback.format_exc(), 'message': '상관관계 분석 중 예상치 못한 오류.' }

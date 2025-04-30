import pandas as pd
import numpy as np
import os
from app.domain.model.crime_schema import Dataset
import traceback

class CrimeCorrelation:
    def __init__(self):
        pass
        
    def analyze_correlation(self, cctv_data, pop_data):
        """CCTV와 인구 데이터의 상관관계를 분석하는 함수"""
        try:
            # 데이터 확인
            if '자치구' in cctv_data.columns and '자치구' in pop_data.columns:
                merge_col = '자치구'
            elif '구별' in cctv_data.columns and '구별' in pop_data.columns:
                merge_col = '구별'
            else:
                # 컬럼명이 일치하지 않는 경우 - 첫 번째 컬럼을 기준으로 rename
                first_col_cctv = cctv_data.columns[0]
                first_col_pop = pop_data.columns[0]
                print(f"컬럼명이 일치하지 않아 첫 번째 컬럼을 기준으로 병합합니다.")
                print(f"CCTV 첫 번째 컬럼: {first_col_cctv}, 인구 첫 번째 컬럼: {first_col_pop}")
                
                # 첫 번째 컬럼을 '자치구'로 통일
                cctv_data = cctv_data.rename(columns={first_col_cctv: '자치구'})
                pop_data = pop_data.rename(columns={first_col_pop: '자치구'})
                merge_col = '자치구'
                
                print(f"컬럼명 변경 후 CCTV 컬럼: {cctv_data.columns.tolist()}")
                print(f"컬럼명 변경 후 인구 컬럼: {pop_data.columns.tolist()}")
            
            print(f"데이터 병합에 사용할 컬럼: {merge_col}")
            
            # 인구 데이터 처리
            pop_data = pop_data.rename(columns={
                pop_data.columns[1]: '인구수',   
                pop_data.columns[2]: '한국인',
                pop_data.columns[3]: '외국인',
                pop_data.columns[4]: '고령자',
            })
            
            # 인덱스 26이 실제로 존재하는지 확인
            if 26 in pop_data.index:
                pop_data.drop([26], inplace=True)
            
            pop_data['외국인비율'] = pop_data['외국인'].astype(int) / pop_data['인구수'].astype(int) * 100
            pop_data['고령자비율'] = pop_data['고령자'].astype(int) / pop_data['인구수'].astype(int) * 100
            
            # 병합
            cctv_pop = pd.merge(cctv_data, pop_data, on=merge_col)
            print(f"병합된 데이터 형태: {cctv_pop.shape}")
            print(f"병합된 데이터 컬럼: {cctv_pop.columns.tolist()}")
            
            # CCTV 개수 컬럼 확인 (소계 또는 다른 이름)
            if '소계' in cctv_pop.columns:
                cctv_col = '소계'
            else:
                # 두 번째 컬럼을 CCTV 개수로 간주
                cctv_col = cctv_data.columns[1]
                print(f"'소계' 컬럼이 없어 {cctv_col}을 CCTV 개수로 사용합니다.")
                # cctv_col 이름 변경
                cctv_pop = cctv_pop.rename(columns={cctv_col: 'CCTV개수'})
                cctv_col = 'CCTV개수'
            
            # 기존 방식: 특정 두 변수 간 상관계수만 계산
            cor1 = np.corrcoef(cctv_pop['고령자비율'], cctv_pop[cctv_col])
            cor2 = np.corrcoef(cctv_pop['외국인비율'], cctv_pop[cctv_col])
            
            print(f'고령자비율과 CCTV의 상관계수: {cor1[0,1]:.4f}')
            print(f'외국인비율과 CCTV의 상관계수: {cor2[0,1]:.4f}')
            
            # 상관계수 해석
            elderly_interpretation = self.get_interpretation_text(cor1[0,1], "고령자비율", "CCTV")
            foreigner_interpretation = self.get_interpretation_text(cor2[0,1], "외국인비율", "CCTV")
            
            print(f"\n{elderly_interpretation}")
            print(f"\n{foreigner_interpretation}")
            
            # 새로운 방식: CCTV와 다른 모든 숫자형 변수들 간의 상관관계만 계산
            print("\n===== CCTV와 다른 변수들 간의 상관관계 =====")
            
            # 데이터프레임에서 숫자형 컬럼만 선택 (CCTV 컬럼 제외)
            numeric_columns = cctv_pop.select_dtypes(include=['int', 'float']).columns.tolist()
            
            # CCTV 컬럼과 다른 숫자형 변수들 간의 상관관계 분석
            cctv_correlations = []
            
            for col in numeric_columns:
                # CCTV 컬럼 자신과의, 또는 비숫자형 컬럼과의 상관관계는 건너뜀
                if col == cctv_col or col == merge_col:
                    continue
                    
                # 상관계수 계산
                corr_value = np.corrcoef(cctv_pop[col], cctv_pop[cctv_col])[0,1]
                # 해석 생성
                interpretation = self.get_interpretation_text(corr_value, col, cctv_col)
                
                print(f"{col} - {cctv_col}: {corr_value:.4f}")
                print(f"  {interpretation}")
                
                cctv_correlations.append({
                    'variable': str(col),
                    'correlation': float(f"{corr_value:.4f}"),
                    'interpretation': interpretation
                })
            
            # 상관계수의 절대값 기준으로 정렬
            cctv_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # 결과 반환
            result = {
                # 기존 결과
                'elderly_correlation': {
                    'value': float(f'{cor1[0,1]:.4f}'),
                    'interpretation': elderly_interpretation
                },
                'foreigner_correlation': {
                    'value': float(f'{cor2[0,1]:.4f}'),
                    'interpretation': foreigner_interpretation
                },
                # 새로운 결과: CCTV와 다른 변수들 간의 상관관계
                'cctv_correlations': cctv_correlations,
                'districts': [
                    {
                        'district': row[merge_col],
                        'cctv_count': int(row[cctv_col]),
                        'population': int(row['인구수']),
                        'elderly_ratio': float(f'{row["고령자비율"]:.2f}'),
                        'foreigner_ratio': float(f'{row["외국인비율"]:.2f}')
                    }
                    for _, row in cctv_pop.iterrows()
                ]
            }
            
            return result
            
        except Exception as e:
            print(f"상관계수 계산 중 오류 발생: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def analyze_crime_correlation(self, cctv_data, crime_data, police_norm_data):
        """CCTV와 범죄 데이터의 상관관계를 분석하는 함수"""
        try:
            print("\n===== CCTV와 범죄 데이터 상관관계 분석 시작 =====\n")
            
            # CCTV 데이터 전처리
            if '자치구' not in cctv_data.columns and cctv_data.columns[0] != '자치구':
                cctv_data = cctv_data.rename(columns={cctv_data.columns[0]: '자치구'})
            
            cctv_col = '소계' if '소계' in cctv_data.columns else cctv_data.columns[1]
            
            # crime_data에서 자치구별 범죄 건수 집계
            if '자치구' not in crime_data.columns:
                # 자치구 컬럼이 없는 경우 마지막 컬럼을 자치구로 가정
                crime_data = crime_data.rename(columns={crime_data.columns[-1]: '자치구'})
            
            # 자치구별로 범죄 건수 집계
            crime_by_district = crime_data.groupby('자치구').agg({
                '살인 발생': 'sum',
                '강도 발생': 'sum',
                '강간 발생': 'sum',
                '절도 발생': 'sum',
                '폭력 발생': 'sum'
            }).reset_index()
            
            # police_norm_data 전처리
            if police_norm_data.columns[0] == '':
                # 첫 번째 컬럼이 비어있는 경우 삭제하고 인덱스 사용
                police_norm_data = police_norm_data.iloc[:, 1:]
                
            # police_norm_data에 자치구 정보가 없으므로 cctv_data의 자치구 순서를 따른다고 가정
            # (실제로는 데이터 정합성 검증이 필요하나 현재 데이터의 순서가 같다고 가정)
            police_norm_data['자치구'] = cctv_data['자치구'].values[:len(police_norm_data)]
            
            print(f"CCTV 데이터 행 수: {len(cctv_data)}")
            print(f"crime_by_district 행 수: {len(crime_by_district)}")
            print(f"police_norm_data 행 수: {len(police_norm_data)}")
            
            # 데이터 병합
            # 1. CCTV와 집계된 범죄 데이터 병합
            merged_data = pd.merge(cctv_data, crime_by_district, on='자치구', how='inner')
            print(f"CCTV + 범죄 데이터 병합 후 행 수: {len(merged_data)}")
            
            # 2. police_norm_data 병합
            merged_data = pd.merge(merged_data, police_norm_data, on='자치구', how='inner')
            print(f"최종 병합 후 행 수: {len(merged_data)}")
            print(f"최종 데이터 컬럼: {merged_data.columns.tolist()}")
            
            # CCTV와 각 범죄 유형별 상관관계 분석
            print("\n===== CCTV와 범죄 유형별 상관관계 =====")
            crime_vars = ['살인', '강도', '강간', '절도', '폭력', '범죄', 
                         '살인검거율', '강도검거율', '강간검거율', '절도검거율', '폭력검거율', '검거']
            
            crime_correlations = []
            
            for col in crime_vars:
                if col not in merged_data.columns:
                    continue
                
                # 상관계수 계산
                corr_value = np.corrcoef(merged_data[col], merged_data[cctv_col])[0,1]
                # 해석 생성
                interpretation = self.get_interpretation_text(corr_value, col, "CCTV")
                
                print(f"{col} - CCTV: {corr_value:.4f}")
                print(f"  {interpretation}")
                
                crime_correlations.append({
                    'variable': str(col),
                    'correlation': float(f"{corr_value:.4f}"),
                    'interpretation': interpretation
                })
            
            # 범죄 발생 건수와의 상관관계도 분석
            for col in ['살인 발생', '강도 발생', '강간 발생', '절도 발생', '폭력 발생']:
                if col not in merged_data.columns:
                    continue
                
                # 상관계수 계산
                corr_value = np.corrcoef(merged_data[col], merged_data[cctv_col])[0,1]
                # 해석 생성
                interpretation = self.get_interpretation_text(corr_value, col, "CCTV")
                
                print(f"{col} - CCTV: {corr_value:.4f}")
                print(f"  {interpretation}")
                
                crime_correlations.append({
                    'variable': str(col),
                    'correlation': float(f"{corr_value:.4f}"),
                    'interpretation': interpretation
                })
            
            # 상관계수의 절대값 기준으로 정렬
            crime_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            print("\n===== CCTV와 범죄 데이터 상관관계 분석 완료 =====\n")
            
            # 결과 반환
            result = {
                'crime_correlations': crime_correlations,
                'districts': [
                    {
                        'district': row['자치구'],
                        'cctv_count': int(row[cctv_col]),
                        'murder': float(row['살인']) if '살인' in row else None,
                        'robbery': float(row['강도']) if '강도' in row else None,
                        'rape': float(row['강간']) if '강간' in row else None,
                        'theft': float(row['절도']) if '절도' in row else None,
                        'violence': float(row['폭력']) if '폭력' in row else None,
                        'crime_index': float(row['범죄']) if '범죄' in row else None,
                        'arrest_rate': float(row['검거']) if '검거' in row else None
                    }
                    for _, row in merged_data.iterrows()
                ]
            }
            
            return result
            
        except Exception as e:
            print(f"범죄 상관계수 계산 중 오류 발생: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def get_interpretation_text(self, corr, var1, var2):
        """상관계수 해석 텍스트를 반환하는 함수"""
        if abs(corr) < 0.1:
            strength = "없음 (매우 약함)"
        elif abs(corr) < 0.3:
            strength = "약함"
        elif abs(corr) < 0.5:
            strength = "중간"
        elif abs(corr) < 0.7:
            strength = "강함"
        else:
            strength = "매우 강함"
            
        direction = "양의" if corr > 0 else "음의"
        
        return f"{var1}과 {var2} 사이에는 {direction} 상관관계가 있으며, 강도는 '{strength}'입니다. {var1}이 증가할 때 {var2}도 {'증가' if corr > 0 else '감소'}하는 경향이 있습니다."

    def load_and_analyze(self, data_dir='app/updated_data'):
        """데이터를 로드하고 상관관계를 분석하는 함수"""
        try:
            print("\n===== 상관계수 분석 시작 =====\n")
            
            # CCTV 데이터 로드
            cctv_file = os.path.join(data_dir, 'cctv_in_seoul.csv')
            cctv_data = pd.read_csv(cctv_file)
            print(f"CCTV 데이터 로드 완료 - 형태: {cctv_data.shape}")
            print(f"CCTV 데이터 컬럼: {cctv_data.columns.tolist()}")
            
            # 인구 데이터 로드
            pop_file = os.path.join(data_dir, 'pop_in_seoul.csv')
            pop_data = pd.read_csv(pop_file)
            print(f"인구 데이터 로드 완료 - 형태: {pop_data.shape}")
            print(f"인구 데이터 컬럼: {pop_data.columns.tolist()}")
            
            # 범죄 데이터 로드
            crime_file = os.path.join(data_dir, 'crime_in_seoul.csv')
            police_norm_file = os.path.join(data_dir, 'police_norm_in_seoul.csv')
            
            try:
                crime_data = pd.read_csv(crime_file)
                police_norm_data = pd.read_csv(police_norm_file)
                print(f"범죄 데이터 로드 완료 - 형태: {crime_data.shape}")
                print(f"경찰서 정규화 데이터 로드 완료 - 형태: {police_norm_data.shape}")
                
                # 범죄 데이터 상관관계 분석
                crime_correlation_results = self.analyze_crime_correlation(cctv_data, crime_data, police_norm_data)
            except Exception as e:
                print(f"범죄 데이터 분석 중 오류 발생: {str(e)}")
                crime_correlation_results = {"error": str(e)}
            
            # 인구 데이터 상관관계 분석
            demographic_results = self.analyze_correlation(cctv_data, pop_data)
            
            # 결과 통합
            results = {
                **demographic_results,
                'crime_analysis': crime_correlation_results
            }
            
            print("\n===== 상관계수 분석 완료 =====\n")
            
            return results
            
        except Exception as e:
            print(f"상관계수 분석 중 오류 발생: {str(e)}")
            print(traceback.format_exc())
            # 오류 발생 시 기본 결과 반환
            return {
                'error': str(e),
                'message': '상관계수 분석 중 오류가 발생했습니다.'
            }

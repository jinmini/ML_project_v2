import os

# 데이터 저장 디렉토리 정의
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
# 디렉토리가 없으면 생성
os.makedirs(save_dir, exist_ok=True)

print(f"데이터 저장 디렉토리: {save_dir}") 
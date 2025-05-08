import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 산출물 저장 경로 설정
# 스크립트가 tf-service/app/test/mnist.py 에 위치한다고 가정하고,
# model/ 과 report/ 디렉토리는 tf-service/ 내에 생성되도록 경로 조정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # tf-service/ 디렉토리
MODEL_DIR = os.path.join(BASE_DIR, 'model')
REPORT_DIR = os.path.join(BASE_DIR, 'report')

# 디렉토리 생성 (존재하지 않는 경우)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

logger.info(f"Base directory set to: {BASE_DIR}")
logger.info(f"Model directory set to: {MODEL_DIR}")
logger.info(f"Report directory set to: {REPORT_DIR}")

def main():
    logger.info("MNIST CNN 분류기 개발 시작")

    # 1. 데이터셋 로딩
    logger.info("MNIST 데이터셋 로딩 중...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    logger.info(f"데이터 로딩 완료. Train: {x_train.shape}, {y_train.shape}, Test: {x_test.shape}, {y_test.shape}")

    # 2. 데이터 전처리
    logger.info("데이터 전처리 시작...")

    # 2.1. 픽셀 정규화 (0~255 -> 0~1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    logger.info("픽셀 정규화 완료.")

    # 2.2. 차원 확장 ((num_samples, 28, 28) -> (num_samples, 28, 28, 1))
    # CNN은 채널 정보를 가진 입력을 기대합니다 (흑백 이미지의 경우 채널 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    logger.info(f"차원 확장 완료. x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

    # 2.3. 레이블 원-핫 인코딩
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    logger.info(f"레이블 원-핫 인코딩 완료. y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    logger.info("데이터 전처리 완료.")

    # 3. 모델 아키텍처 정의 (CNN)
    logger.info("CNN 모델 아키텍처 정의 시작...")
    model = Sequential([
        # 첫 번째 Convolutional Layer + MaxPooling Layer
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
        MaxPooling2D(pool_size=(2, 2)),
        
        # 두 번째 Convolutional Layer + MaxPooling Layer
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten Layer: 2D 특징 맵을 1D 벡터로 변환
        Flatten(),
        
        # Dense Layer (Fully Connected Layer)
        Dense(128, activation='relu'),
        
        # 출력 Layer: 10개의 클래스에 대한 확률 출력
        Dense(num_classes, activation='softmax')
    ])
    logger.info("CNN 모델 아키텍처 정의 완료.")

    # 모델 구조 요약 출력
    model.summary(print_fn=logger.info)

    # 4. 모델 컴파일
    logger.info("모델 컴파일 시작...")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    logger.info("모델 컴파일 완료.")

    # 5. 모델 학습 실행
    logger.info("모델 학습 시작...")
    epochs = 10
    batch_size = 128
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2, # 학습 데이터의 20%를 검증 데이터로 사용
                        verbose=1) # 학습 진행 상황 출력
    logger.info("모델 학습 완료.")
    logger.info(f"학습 히스토리 키: {history.history.keys()}")

    # 6. 학습 과정 시각화 및 저장
    logger.info("학습 과정 시각화 시작...")
    
    # 정확도 그래프
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_plot_path = os.path.join(REPORT_DIR, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)
    logger.info(f"정확도 그래프 저장 완료: {accuracy_plot_path}")
    plt.close() # 이전 그림 닫기

    # 손실 그래프
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1) # 실제로는 subplot(1,1,1)과 같음. 필요시 1,2,2로 다른 그래프 추가 가능
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(REPORT_DIR, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    logger.info(f"손실 그래프 저장 완료: {loss_plot_path}")
    plt.close() # 이전 그림 닫기
    
    logger.info("학습 과정 시각화 완료.")

    # 7. 모델 평가
    logger.info("모델 평가 시작...")
    score = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f'Test loss: {score[0]}')
    logger.info(f'Test accuracy: {score[1]}')
    logger.info("모델 평가 완료.")

    # 8. 모델 저장
    logger.info("모델 저장 시작...")
    model_path = os.path.join(MODEL_DIR, 'mnist_cnn_model.h5')
    try:
        model.save(model_path)
        logger.info(f"모델이 성공적으로 저장되었습니다: {model_path}")
    except Exception as e:
        logger.error(f"모델 저장 중 오류 발생: {e}")
    logger.info("모델 저장 완료.")

    logger.info("MNIST CNN 분류기 개발 프로세스 완료.")

if __name__ == '__main__':
    main()

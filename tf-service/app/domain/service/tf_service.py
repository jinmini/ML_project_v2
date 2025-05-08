import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import logging
from tensorflow.keras.models import load_model

# 로거 설정
logger = logging.getLogger(__name__)

# 저장 경로 설정
UPLOAD_DIR = "./uploads"
MOSAIC_DIR = "./mosaic"

# 얼굴 감지를 위한 Haar Cascade 분류기 로드
try:
    # OpenCV에서 제공하는 사전 훈련된 얼굴 감지 모델 사용
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logger.info("얼굴 감지 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    logger.error(f"얼굴 감지 모델 로드 실패: {e}")
    face_cascade = None

# 이미지 인식을 위한 MobileNetV2 모델 로드 (필요시)
image_model = None

# --- MNIST Prediction Specifics ---
# Construct the model path relative to this file's location
# This file: tf-service/app/domain/service/tf_service.py
# Target model: tf-service/model/mnist_cnn_model.h5
_SERVICE_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_TF_SERVICE_ROOT_DIR = os.path.abspath(os.path.join(_SERVICE_FILE_DIR, '..', '..', '..')) # Should be tf-service/
MNIST_MODEL_PATH = os.path.join(_TF_SERVICE_ROOT_DIR, 'model', 'mnist_cnn_model.h5')

mnist_model_global = None # Global variable to store the loaded MNIST model

def load_mnist_model_globally():
    """
    Loads the pre-trained MNIST CNN model into a global variable.
    Should be called once, ideally at application startup.
    Returns True if successful or already loaded, False otherwise.
    """
    global mnist_model_global
    if mnist_model_global is None:
        if not os.path.exists(MNIST_MODEL_PATH):
            logger.error(f"MNIST model file not found at: {MNIST_MODEL_PATH}")
            return False
        try:
            # Ensure TensorFlow is available here. It's imported as 'tf' in the provided file.
            # Explicitly use tf.keras.models.load_model if 'load_model' itself isn't directly imported.
            mnist_model_global = tf.keras.models.load_model(MNIST_MODEL_PATH)
            logger.info(f"MNIST CNN model loaded successfully from {MNIST_MODEL_PATH}.")
            return True
        except Exception as e:
            logger.error(f"Error loading MNIST CNN model from {MNIST_MODEL_PATH}: {e}")
            mnist_model_global = None # Ensure it's reset on failure
            return False
    return True # Already loaded

def predict_mnist_digit(image_path: str):
    """
    Predicts a digit from an image file using the globally loaded MNIST model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary with "success" (bool), "predicted_digit" (int or None),
              and "message" (str).
    """
    global mnist_model_global
    if mnist_model_global is None:
        logger.warning("MNIST model not loaded. Attempting to load now for prediction.")
        if not load_mnist_model_globally():
            return {"success": False, "predicted_digit": None, "message": "MNIST model is not available or failed to load."}

    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file for MNIST prediction not found: {image_path}")
            return {"success": False, "predicted_digit": None, "message": f"Image file not found: {os.path.basename(image_path)}"}

        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Could not read image (or image is empty) from path: {image_path}")
            return {"success": False, "predicted_digit": None, "message": f"Could not read image data: {os.path.basename(image_path)}"}

        # Preprocessing for MNIST model: Resize to 28x28
        img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to 0-1
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Reshape for model input: (1, 28, 28, 1)
        img_reshaped = img_normalized.reshape(1, 28, 28, 1)

        # Perform prediction
        predictions = mnist_model_global.predict(img_reshaped)
        predicted_digit = int(np.argmax(predictions[0])) # Get the class with highest probability

        logger.info(f"Predicted MNIST digit for '{os.path.basename(image_path)}' is: {predicted_digit}")
        return {"success": True, "predicted_digit": predicted_digit, "message": "Prediction successful."}

    except Exception as e:
        logger.error(f"Error during MNIST prediction for '{image_path}': {e}")
        # For more detailed debugging in development, you might want to log the traceback:
        # import traceback
        # logger.error(traceback.format_exc())
        return {"success": False, "predicted_digit": None, "message": f"An error occurred during prediction: {str(e)}"}

# Consider calling load_mnist_model_globally() once when the service starts.
# This can be done in the main application setup (e.g., using FastAPI's lifespan manager).
# For now, it's lazy-loaded on the first prediction attempt if not already loaded.

# --- END MNIST Prediction Specifics ---

def load_image_model():
    """이미지 인식 모델을 지연 로딩하는 함수"""
    global image_model
    if image_model is None:
        try:
            # 가중치 사전 로드된 MobileNetV2 모델 사용
            image_model = MobileNetV2(weights='imagenet', include_top=True)
            logger.info("이미지 인식 모델이 성공적으로 로드되었습니다.")
        except Exception as e:
            logger.error(f"이미지 인식 모델 로드 실패: {e}")
            return None
    return image_model

def detect_faces(image_path):
    """
    이미지에서 얼굴을 감지하는 함수
    
    Args:
        image_path (str): 이미지 파일의 경로
        
    Returns:
        tuple: (이미지, 얼굴 좌표 목록) 반환
    """
    try:
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"이미지를 읽을 수 없음: {image_path}")
            return None, []
        
        # 얼굴 감지를 위해 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        logger.info(f"이미지에서 {len(faces)}개의 얼굴이 감지되었습니다.")
        return img, faces
    except Exception as e:
        logger.error(f"얼굴 감지 중 오류 발생: {e}")
        return None, []

def apply_mosaic(img, faces, mosaic_size=15):
    """
    감지된 얼굴에 모자이크 처리를 적용하는 함수
    
    Args:
        img (numpy.ndarray): 원본 이미지
        faces (list): 얼굴 좌표 목록 (x, y, w, h)
        mosaic_size (int): 모자이크 블록 크기
        
    Returns:
        numpy.ndarray: 모자이크 처리된 이미지
    """
    try:
        # 원본 이미지 복사
        mosaic_img = img.copy()
        
        # 각 얼굴에 모자이크 적용
        for (x, y, w, h) in faces:
            # 얼굴 영역 추출
            face_region = mosaic_img[y:y+h, x:x+w]
            
            # 축소 (픽셀화 효과를 위해)
            small = cv2.resize(face_region, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
            
            # 원래 크기로 확대 (픽셀화된 효과 생성)
            mosaic_face = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 원본 이미지에 모자이크 처리된 얼굴 적용
            mosaic_img[y:y+h, x:x+w] = mosaic_face
        
        logger.info("모자이크 처리가 성공적으로 적용되었습니다.")
        return mosaic_img
    except Exception as e:
        logger.error(f"모자이크 처리 중 오류 발생: {e}")
        return img

def recognize_image(image_path, top_k=5):
    """
    이미지의 내용을 인식하는 함수
    
    Args:
        image_path (str): 이미지 파일의 경로
        top_k (int): 반환할 상위 예측 수
        
    Returns:
        list: 예측 결과 목록 [(레이블, 설명, 확률), ...]
    """
    try:
        # 이미지 인식 모델 로드
        model = load_image_model()
        if model is None:
            return []
        
        # 이미지 로드 및 전처리
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img = cv2.resize(img, (224, 224))  # MobileNetV2 입력 크기
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # 예측
        preds = model.predict(img)
        results = decode_predictions(preds, top=top_k)[0]
        
        # 결과 포맷팅
        formatted_results = [
            {"label": label, "description": desc, "probability": float(prob)}
            for (_, label, desc, prob) in [(None, *r) for r in results]
        ]
        
        logger.info(f"이미지 인식 결과: {formatted_results}")
        return formatted_results
    except Exception as e:
        logger.error(f"이미지 인식 중 오류 발생: {e}")
        return []

def process_image(image_path, mosaic_size=15):
    """
    이미지 처리 전체 프로세스를 실행하는 함수
    
    Args:
        image_path (str): 이미지 파일의 경로
        mosaic_size (int): 모자이크 블록 크기
        
    Returns:
        dict: 처리 결과 정보
    """
    try:
        # 디렉토리가 없으면 생성
        if not os.path.exists(MOSAIC_DIR):
            os.makedirs(MOSAIC_DIR)
            logger.info(f"'{MOSAIC_DIR}' 디렉토리를 생성했습니다.")
        
        # 얼굴 감지
        img, faces = detect_faces(image_path)
        if img is None:
            return {"success": False, "message": "이미지를 읽을 수 없습니다.", "faces_detected": 0}
        
        # 모자이크 처리
        if len(faces) > 0:
            mosaic_img = apply_mosaic(img, faces, mosaic_size)
            
            # 원본 파일명에서 확장자 분리
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            
            # 모자이크 처리된 이미지 저장
            mosaic_path = os.path.join(MOSAIC_DIR, f"{name}_mosaic{ext}")
            cv2.imwrite(mosaic_path, mosaic_img)
            logger.info(f"모자이크 처리된 이미지가 '{mosaic_path}' 경로에 저장되었습니다.")
            
            # 이미지 인식 (필요시)
            recognition_results = recognize_image(image_path)
            
            return {
                "success": True,
                "message": "이미지 처리 성공",
                "original_image": image_path,
                "mosaic_image": mosaic_path,
                "faces_detected": len(faces),
                "recognition_results": recognition_results
            }
        else:
            logger.info("감지된 얼굴이 없습니다.")
            return {
                "success": True,
                "message": "감지된 얼굴이 없습니다.",
                "original_image": image_path,
                "faces_detected": 0
            }
    except Exception as e:
        logger.error(f"이미지 처리 중 오류 발생: {e}")
        return {"success": False, "message": f"이미지 처리 실패: {str(e)}", "faces_detected": 0}

# app.py
import os
import io
import numpy as np
from PIL import Image
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = FastAPI()

# 1. 특징 추출 모델 로드 (ResNet50)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 2. 로컬 이미지 디렉토리 설정
images_dir = r"C:\Users\kang\Desktop\EC\images"

# 3. 데이터베이스 이미지의 특징 벡터 미리 추출
def extract_features(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

database_features = []
database_records = []

# 로컬 디렉토리의 모든 이미지 로드 및 특징 추출
def load_database_images():
    global database_features, database_records
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(images_dir, filename)
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((224, 224))
                features = extract_features(img, model)
                database_features.append(features)
                database_records.append({
                    'image_name': filename,
                    'image_path': image_path
                })
            except Exception as e:
                print(f"이미지 {filename} 로드 실패: {e}")

# 서버 시작 시 이미지 로드
@app.on_event("startup")
def startup_event():
    load_database_images()
    print(f"로컬 디렉토리 '{images_dir}'에서 {len(database_records)}개의 이미지를 로드했습니다.")

# 4. 유사한 이미지 찾기 함수
def find_similar_images(query_img, model, database_features, database_records, top=5, threshold=0.85):
    query_features = extract_features(query_img, model)
    similarities = cosine_similarity([query_features], database_features)[0]
    max_similarity = np.max(similarities)
    max_index = np.argmax(similarities)

    if max_similarity >= threshold:
        # 유사도가 임계값 이상인 경우
        similar_images = [database_records[max_index]]
        message = "추천 상품입니다."
    else:
        # 유사도가 임계값 미만인 경우, 상위 N개 이미지 반환
        similar_indices = similarities.argsort()[-top:][::-1]
        similar_images = [database_records[i] for i in similar_indices]
        message = "이런 상품들은 어떠세요?"
    return similar_images, message

# 5. 이미지 URL로부터 이미지를 로드하는 함수
def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img = img.resize((224, 224))
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {e}")

# 6. FastAPI 엔드포인트 정의
@app.post("/find_similar")
async def find_similar_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    if file is None and image_url is None:
        raise HTTPException(status_code=400, detail="이미지 파일이나 URL을 제공해야 합니다.")

    # 업로드된 이미지 열기
    if file is not None:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            img = img.resize((224, 224))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {e}")
    elif image_url is not None:
        img = load_image_from_url(image_url)
    else:
        raise HTTPException(status_code=400, detail="이미지 로드 실패: 파일이나 URL이 유효하지 않습니다.")

    # 유사한 이미지 찾기
    similar_imgs, msg = find_similar_images(
        img, model, database_features, database_records, top=5, threshold=0.85
    )

    # 결과 준비
    result_images = []
    for record in similar_imgs:
        result_images.append({
            "image_name": record['image_name'],
            "image_path": record['image_path']
        })

    return JSONResponse(content={"message": msg, "similar_images": result_images})

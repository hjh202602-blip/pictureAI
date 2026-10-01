import io
import json
import os
from pathlib import Path
import re
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
from PIL import Image

# 1. 환경 변수 및 절대 경로 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("key") or os.getenv("GEMINI_API_KEY")
ADMIN_KEY_PATH = os.getenv("admin", "serviceAccountKey.json")

# 실행 위치와 상관없이 templates 폴더를 정확히 찾도록 고정
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

# 2. Firebase 초기화
if not firebase_admin._apps:
    if not os.path.exists(ADMIN_KEY_PATH):
        raise RuntimeError(f"❌ Firebase 키 파일을 찾을 수 없습니다: '{ADMIN_KEY_PATH}'")
    try:
        cred = credentials.Certificate(ADMIN_KEY_PATH)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        raise RuntimeError(f"❌ Firebase 인증 실패: {e}")

db = firestore.client()

# 3. FastAPI 앱 생성
app = FastAPI(title="AI Price Lens & Ranking")


# 4. 웹 페이지 라우트 (Jinja2 충돌 없는 안전한 로딩 방식)
@app.get("/", response_class=HTMLResponse)
async def login_page():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse(
            f"❌ '{index_file}' 경로에 파일이 없습니다. 폴더 구조를 확인하세요.", 
            status_code=404
        )
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.post("/start", response_class=HTMLResponse)
async def start_camera_page(username: str = Form(...)):
    camera_file = TEMPLATES_DIR / "camera.html"
    if not camera_file.exists():
        return HTMLResponse(
            f"❌ '{camera_file}' 경로에 파일이 없습니다.", 
            status_code=404
        )
    html_content = camera_file.read_text(encoding="utf-8")
    # {{ username }} 치환
    html_content = html_content.replace("{{ username }}", username)
    return HTMLResponse(content=html_content)


# 5. 실시간 TOP 10 랭킹 API
@app.get("/api/rankings")
async def get_rankings():
    try:
        docs = (
            db.collection("rankings")
            .order_by("price_num", direction=firestore.Query.DESCENDING)
            .limit(10)
            .stream()
        )

        rankings = []
        for rank, doc in enumerate(docs, start=1):
            item = doc.to_dict()
            rankings.append({
                "rank": rank,
                "username": item.get("username", "익명"),
                "name": item.get("name", ""),
                "tag": item.get("tag", ""),
                "price": item.get("price", "0원"),
                "price_num": item.get("price_num", 0),
                "reason": item.get("reason", "")
            })
        return JSONResponse(content=rankings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"랭킹 조회 실패: {e}")


# 6. 이미지 감정 및 랭킹 저장 API
@app.post("/evaluate")
async def evaluate_image(
    username: str = Form(...),
    file: UploadFile = File(...)
):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key가 누락되었습니다.")

    try:
        # 이미지 최적화 (512px 초경량)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = """
        사진 속 대상을 식별하고 대략적인 가격이나 가치를 유쾌하고 빠르게 판정해줘.

        [판정 룰 - 엄격 준수]
        1. 사람/얼굴:
           - 아무리 잘생기고 예뻐도 절대 200만 원 초과 금지 (MAX 2,000,000원 제한).
           - 현실감 있고 킹받는 생활 밀착형 시세 책정 (예: 훈남상 180만 원, 직장인 4,500원).
        2. 일반 물건: 시중/중고 체감 시세
        3. 반려동물: 츄르 환산 가치 (최대 200만 원 선)
        4. 음식/기타: 현실적인 식당/배달 시세

        반드시 다음 JSON 형식으로만 출력하고 정수형 price_num 필드를 포함할 것:
        {
          "name": "식별 대상 요약 (예: 카페 훈남 알바상)",
          "tag": "카테고리 태그 (예: 훈남 상한가)",
          "price": "180만 원",
          "price_num": 1800000,
          "reason": "왜 이 가격인지 한 줄 이유"
        }
        """

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[image, prompt],
            config={"temperature": 0.7}
        )

        raw_text = response.text.strip()
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        data = json.loads(match.group(0)) if match else json.loads(raw_text)

        price_num = int(data.get("price_num", 0))
        if "사람" in data.get("tag", "") or "관상" in data.get("name", ""):
            if price_num > 2000000:
                price_num = 2000000
                data["price"] = "200만 원 (상한선)"
                data["price_num"] = 2000000

        # Firestore에 저장
        record = {
            "username": username,
            "name": data.get("name", "미확인"),
            "tag": data.get("tag", "미분류"),
            "price": data.get("price", "0원"),
            "price_num": price_num,
            "reason": data.get("reason", ""),
            "created_at": datetime.utcnow()
        }
        db.collection("rankings").add(record)

        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import io
import json
import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from google import genai
from PIL import Image

# .env 파일에서 key=... 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("key") or os.getenv("GEMINI_API_KEY")

app = FastAPI(title="AI Price Lens")

# ==========================================
# 1. 로그인 화면 HTML
# ==========================================
LOGIN_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>아이디 입력</title>
    <style>
        body {
            margin: 0;
            padding: 30px;
            text-align: center;
            font-family: sans-serif;
            background: #f5f5f5;
        }
        h1 {
            margin-top: 80px;
            margin-bottom: 40px;
        }
        input {
            width: 80%;
            max-width: 400px;
            padding: 15px;
            font-size: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
            box-sizing: border-box;
        }
        button {
            margin-top: 20px;
            padding: 15px 40px;
            font-size: 20px;
            border: none;
            border-radius: 10px;
            background: #222;
            color: white;
            cursor: pointer;
        }
        button:active {
            transform: scale(0.97);
        }
    </style>
</head>
<body>
    <h1>👤 아이디를 입력하세요</h1>
    <form method="POST" action="/start">
        <input
            type="text"
            name="username"
            placeholder="아이디 입력"
            maxlength="20"
            required
            autocomplete="off"
        >
        <br>
        <button type="submit">시작하기</button>
    </form>
</body>
</html>
"""

# ==========================================
# 2. 실시간 카메라 연동 화면 HTML
# ==========================================
CAMERA_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>사진 촬영</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            text-align: center;
            font-family: sans-serif;
            background: #f5f5f5;
        }
        h1 {
            margin-bottom: 5px;
        }
        .username {
            font-size: 18px;
            color: #555;
            margin-bottom: 25px;
        }
        .btn {
            display: inline-block;
            padding: 16px 36px;
            background: #222;
            color: white;
            border-radius: 12px;
            font-size: 20px;
            font-weight: bold;
            cursor: pointer;
            border: none;
            margin: 5px;
        }
        .btn:active {
            transform: scale(0.97);
        }
        .btn-capture {
            background: #e53e3e;
            display: none;
        }
        /* 실시간 비디오 및 캡처 프리뷰 */
        #video-container {
            margin: 20px auto;
            max-width: 450px;
            position: relative;
        }
        #video, #preview {
            width: 100%;
            max-height: 420px;
            border-radius: 15px;
            background: #000;
            object-fit: cover;
            display: none;
        }
        /* 숨겨진 캡처용 캔버스 */
        #canvas {
            display: none;
        }
        /* 결과 카드 */
        #result-box {
            display: none;
            margin: 25px auto;
            max-width: 400px;
            background: #fff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            text-align: center;
        }
        .badge {
            display: inline-block;
            background: #eee;
            color: #333;
            font-size: 14px;
            font-weight: bold;
            padding: 4px 10px;
            border-radius: 20px;
            margin-bottom: 10px;
        }
        .item-name {
            font-size: 18px;
            font-weight: bold;
            color: #222;
            margin-bottom: 8px;
        }
        .price {
            font-size: 32px;
            font-weight: 900;
            color: #e53e3e;
            margin: 10px 0;
        }
        .reason {
            font-size: 14px;
            color: #666;
            line-height: 1.4;
            background: #fafafa;
            padding: 10px;
            border-radius: 8px;
        }
        #loading {
            display: none;
            font-size: 18px;
            font-weight: bold;
            color: #222;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>📷 사진 촬영</h1>
    <div class="username">아이디: <b>{{ username }}</b></div>

    <div>
        <button id="btn-start" class="btn" onclick="startCamera()">📷 카메라 켜기</button>
        <button id="btn-snap" class="btn btn-capture" onclick="takeSnapshot()">📸 찰칵! 촬영하기</button>
    </div>

    <div id="loading">⚡ AI가 순식간에 감정 중...</div>

    <div id="video-container">
        <!-- 실시간 카메라 화면 -->
        <video id="video" autoplay playsinline></video>
        <!-- 촬영된 사진 프리뷰 -->
        <img id="preview">
    </div>

    <canvas id="canvas"></canvas>

    <!-- 가격 감정 결과 박스 -->
    <div id="result-box">
        <span class="badge" id="res-tag">#감정완료</span>
        <div class="item-name" id="res-name">식별 대상</div>
        <div class="price" id="res-price">0원</div>
        <div class="reason" id="res-reason">이유</div>
    </div>

    <script>
        const video = document.getElementById("video");
        const canvas = document.getElementById("canvas");
        const preview = document.getElementById("preview");
        const btnStart = document.getElementById("btn-start");
        const btnSnap = document.getElementById("btn-snap");
        const loading = document.getElementById("loading");
        const resultBox = document.getElementById("result-box");
        let stream = null;

        // 1. 카메라 권한 요청 및 실시간 화면 띄우기
        async function startCamera() {
            try {
                // 후면 카메라 우선, 미지원 시 기본 웹캠 실행
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
                    audio: false
                });
                video.srcObject = stream;
                video.style.display = "block";
                preview.style.display = "none";
                resultBox.style.display = "none";

                btnStart.style.display = "none";
                btnSnap.style.display = "inline-block";
            } catch (err) {
                alert("카메라를 켤 수 없습니다. 브라우저 카메라 권한을 확인해주세요.");
            }
        }

        // 2. 사진 캡처 및 AI 서버 전송
        function takeSnapshot() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // 실시간 비디오 끄고 캡처된 정지 화면 표시
            preview.src = canvas.toDataURL("image/jpeg");
            preview.style.display = "block";
            video.style.display = "none";
            btnSnap.style.display = "none";
            btnStart.style.display = "inline-block";
            btnStart.innerText = "🔄 다시 찍기";

            // 카메라 리소스 해제
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }

            loading.style.display = "block";
            resultBox.style.display = "none";

            // 이미지를 Blob으로 변환해 백엔드 API 호출
            canvas.toBlob(async function(blob) {
                const formData = new FormData();
                formData.append("file", blob, "photo.jpg");

                try {
                    const response = await fetch("/evaluate", {
                        method: "POST",
                        body: formData
                    });
                    const data = await response.json();
                    loading.style.display = "none";

                    if (response.ok) {
                        document.getElementById("res-tag").innerText = "#" + data.tag;
                        document.getElementById("res-name").innerText = data.name;
                        document.getElementById("res-price").innerText = data.price;
                        document.getElementById("res-reason").innerText = "💬 " + data.reason;
                        resultBox.style.display = "block";
                    } else {
                        alert("분석 실패: " + (data.detail || "알 수 없는 오류"));
                    }
                } catch (e) {
                    loading.style.display = "none";
                    alert("서버 통신 오류가 발생했습니다.");
                }
            }, "image/jpeg", 0.85);
        }
    </script>
</body>
</html>
"""

# ==========================================
# 3. 백엔드 API 라우트
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def login_page():
    return LOGIN_HTML

@app.post("/start", response_class=HTMLResponse)
async def start_camera_page(username: str = Form(...)):
    return HTMLResponse(content=CAMERA_HTML.replace("{{ username }}", username))

@app.post("/evaluate")
async def evaluate_image(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key가 누락되었습니다.")

    try:
        # 이미지 로드 및 512px 초경량 압축 (초고속 처리용)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = """
        사진 속 대상을 보고 대략적인 가격이나 가치를 유쾌하고 빠르게 판정해줘.

        [판정 룰 - 엄격 준수]
        1. 사람/얼굴:
           - 아무리 연예인급, 국보급 외모라도 절대 200만 원을 넘기지 말 것 (MAX 200만 원 제한).
           - 현실감 있고 킹받는 생활 밀착형 시세로 책정할 것.
           - 예시:
             * 역대급 훈남/훈녀 관상: 180만~200만 원
             * 호감형 카페 알바생 관상: 65만 원
             * 주말에 같이 피시방 갈 친구 관상: 15,000원
             * 그냥 흔한 피곤한 직장인/학생: 4,500원 (아메리카노 한 잔 값)
        2. 일반 물건/상품: 대략적인 중고/온라인 체감 시세
        3. 반려동물: 츄르 5,000개 값 (또는 100만~200만 원 선에서 귀여움 수치로 환산)
        4. 음식/기타: 현실적인 식당/배달 시세

        반드시 다음 JSON 규격으로만 응답해:
        {
          "name": "식별 대상 요약 (예: 훈남 카페 알바상, 찌든 월요일 직장인 관상, 에어팟 프로)",
          "tag": "카테고리 태그 (예: 훈남 상한가, 피시방 메이트, 가성비 관상)",
          "price": "예상 금액 (예: 185만 원, 8,000원, 45,000원)",
          "reason": "왜 이 가격인지 납득가는 유쾌한 한 줄 이유"
        }
        """

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[image, prompt],
            config={"temperature": 0.7}
        )

        raw_text = response.text.strip()
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
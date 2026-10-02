import hashlib
import html
import io
import math
import os
import random

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

app = FastAPI(title="AI Price Lens")

MODEL_ID = os.getenv("MODEL_ID", "openai/clip-vit-large-patch14")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
if DEVICE == "cuda":
    model.half()
processor = CLIPProcessor.from_pretrained(MODEL_ID)

MAX_UPLOAD = 10 * 1024 * 1024
MIN_CONFIDENCE = 0.12

OBJECTS = [
    ("스마트폰",   "a smartphone",              (200_000, 1_500_000),   "액정 상태가 시세를 좌우해요"),
    ("노트북",     "a laptop computer",          (400_000, 2_500_000),   "스펙보다 배터리 상태가 중요해요"),
    ("태블릿",     "a tablet computer",          (200_000, 1_200_000),   "저장 용량에 따라 가격 차이가 커요"),
    ("이어폰",     "wireless earbuds",           (15_000, 300_000),      "한쪽만 있으면 가치가 뚝 떨어져요"),
    ("헤드폰",     "over-ear headphones",        (30_000, 450_000),      "노이즈캔슬링이면 프리미엄이에요"),
    ("키보드",     "a computer keyboard",        (10_000, 200_000),      "기계식이면 가격이 올라가요"),
    ("마우스",     "a computer mouse",           (5_000, 120_000),       "무선·게이밍이면 비싸져요"),
    ("모니터",     "a computer monitor",         (100_000, 600_000),     "해상도와 주사율이 가격을 정해요"),
    ("카메라",     "a camera",                   (150_000, 2_000_000),   "셔터 수가 적을수록 값이 나가요"),
    ("손목시계",   "a wristwatch",               (30_000, 1_500_000),    "브랜드에 따라 천차만별이에요"),
    ("안경",       "eyeglasses",                 (30_000, 300_000),      "렌즈 가격이 따로 붙어요"),
    ("선글라스",   "sunglasses",                 (20_000, 300_000),      "브랜드 값이 대부분이에요"),
    ("가방",       "a bag or backpack",          (20_000, 500_000),      "브랜드와 소재에 따라 달라요"),
    ("지갑",       "a wallet",                   (15_000, 400_000),      "가죽 종류가 가격을 좌우해요"),
    ("열쇠",       "a set of keys",              (1_000, 10_000),        "복사 비용 정도예요"),
    ("신발",       "a pair of shoes",            (30_000, 250_000),      "한정판이면 훨씬 비싸요"),
    ("모자",       "a hat or cap",               (10_000, 80_000),       "브랜드 로고 유무로 갈려요"),
    ("옷",         "a piece of clothing",        (15_000, 200_000),      "소재와 브랜드에 따라 달라요"),
    ("우산",       "an umbrella",                (5_000, 40_000),        "자동 우산이면 조금 더 해요"),
    ("텀블러",     "a tumbler or water bottle",  (8_000, 50_000),        "보온·보냉 성능이 가격을 정해요"),
    ("머그컵",     "a mug or cup",               (3_000, 20_000),        "디자인 컵이면 더 비싸요"),
    ("커피",       "a cup of coffee",            (2_000, 7_000),         "프랜차이즈 기준 가격이에요"),
    ("책",         "a book",                     (8_000, 30_000),        "신간 정가 기준이에요"),
    ("공책",       "a notebook",                 (1_000, 8_000),         "스프링 노트가 조금 더 해요"),
    ("펜",         "a pen",                      (500, 5_000),           "고급 만년필은 별개예요"),
    ("가위",       "scissors",                   (2_000, 15_000),        "잘 드는 정도가 포인트예요"),
    ("리모컨",     "a remote control",           (3_000, 30_000),        "범용 리모컨 기준이에요"),
    ("충전기",     "a phone charger cable",      (5_000, 50_000),        "급속 충전 지원이면 더 비싸요"),
    ("보조배터리", "a power bank",               (10_000, 60_000),       "용량이 클수록 비싸요"),
    ("의자",       "a chair",                    (30_000, 400_000),      "사무용 메쉬 의자는 더 비싸요"),
    ("조명",       "a lamp",                     (10_000, 150_000),      "인테리어 조명은 더 해요"),
    ("거울",       "a mirror",                   (5_000, 100_000),       "크기가 가격을 정해요"),
    ("화분",       "a houseplant in a pot",      (3_000, 50_000),        "식물 종류에 따라 달라요"),
    ("꽃다발",     "a bouquet of flowers",       (10_000, 60_000),       "시즌에 따라 변동이 커요"),
    ("인형",       "a stuffed toy",              (5_000, 60_000),        "캐릭터 정품이면 더 비싸요"),
    ("장난감",     "a toy",                      (5_000, 80_000),        "한정판은 가격이 뛰어요"),
    ("게임기",     "a game console",             (200_000, 700_000),     "번들 구성에 따라 달라요"),
    ("공",         "a ball",                     (5_000, 40_000),        "공인구는 더 비싸요"),
    ("자전거",     "a bicycle",                  (100_000, 1_000_000),   "프레임 소재가 가격을 정해요"),
    ("자동차",     "a car",                      (3_000_000, 40_000_000), "연식과 주행거리가 핵심이에요"),
    ("빵",         "bread or a pastry",          (1_500, 6_000),         "베이커리 기준이에요"),
    ("과일",       "fruit",                      (2_000, 15_000),        "제철이면 저렴해요"),
    ("라면",       "a bowl of instant noodles",  (1_000, 5_000),         "봉지 라면 기준이에요"),
    ("피자",       "a pizza",                    (12_000, 30_000),       "사이즈에 따라 달라요"),
    ("치킨",       "fried chicken",              (15_000, 25_000),       "배달비는 별도예요"),
    ("햄버거",     "a hamburger",                (4_000, 10_000),        "세트로 바꾸면 추가돼요"),
    ("과자",       "a bag of snacks",            (1_000, 3_000),         "편의점 기준이에요"),
    ("음료캔",     "a soda can",                 (1_000, 2_500),         "편의점 기준이에요"),
    ("물병",       "a plastic water bottle",     (500, 2_000),           "마트 기준이에요"),
    ("음식",       "a plate of food",            (6_000, 20_000),        "식당 한 끼 기준이에요"),
]

NON_OBJECTS = [
    ("사람",     "a person or a face",             "사람은 비매품이에요. 물건을 비춰주세요"),
    ("반려동물", "a pet animal",                   "반려동물은 가족이라 비매품이에요"),
    ("풍경",     "a landscape or scenery",         "풍경은 값을 매길 수 없어요. 물건을 비춰주세요"),
    ("빈공간",   "an empty room or a plain wall",  "물건이 보이지 않아요. 가까이서 찍어보세요"),
]

N_O = len(OBJECTS)
ALL_TEXTS = [o[1] for o in OBJECTS] + [n[1] for n in NON_OBJECTS]

TEMPLATES = [
    "a photo of {}.",
    "a close-up photo of {}.",
    "a blurry photo of {}.",
    "a low resolution photo of {}.",
    "a bright photo of {}.",
    "a photo taken with a phone, showing {}.",
    "an image containing {}.",
    "a snapshot of {}.",
]
TTA_VIEWS = int(os.getenv("TTA_VIEWS", "4"))


def _tensor(out, *names):
    if isinstance(out, torch.Tensor):
        return out
    for n in (*names, "pooler_output"):
        v = getattr(out, n, None)
        if v is not None:
            return v
    raise RuntimeError("특징 벡터를 읽을 수 없습니다 (transformers 버전 확인)")


@torch.no_grad()
def encode_images(images):
    pv = processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE, dtype=model.dtype)
    f = _tensor(model.get_image_features(pixel_values=pv), "image_embeds")
    return F.normalize(f.float(), dim=-1).cpu()


@torch.no_grad()
def build_text_weights():
    rows = []
    for text in ALL_TEXTS:
        tok = processor(text=[t.format(text) for t in TEMPLATES], return_tensors="pt", padding=True)
        tok = {k: v.to(DEVICE) for k, v in tok.items()}
        f = _tensor(model.get_text_features(**tok), "text_embeds")
        f = F.normalize(f.float(), dim=-1).mean(0)
        rows.append(F.normalize(f, dim=-1).cpu())
    return torch.stack(rows)


print("텍스트 임베딩 계산 중...")
W = build_text_weights()
LOGIT_SCALE = float(model.logit_scale.exp().item())


def make_views(image):
    w, h = image.size
    s = max(w, h)
    cw, ch = int(w * 0.7), int(h * 0.7)
    l, t = (w - cw) // 2, (h - ch) // 2
    views = [
        image,
        ImageOps.mirror(image),
        ImageOps.pad(image, (s, s), color=(124, 116, 104)),
        image.crop((l, t, l + cw, t + ch)),
    ]
    return views[:max(1, TTA_VIEWS)]


def score_image(image):
    f = F.normalize(encode_images(make_views(image)).mean(0), dim=-1)
    return (LOGIT_SCALE * (f @ W.T)).softmax(0)


def roll_price(lo, hi, rng):
    price = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    step = 100 if price < 100_000 else 1_000 if price < 10_000_000 else 10_000
    return max(step, int(price) // step * step)


LOGIN_HTML = """

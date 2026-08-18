# ♻️ 재활용품 VQA (Visual Question Answering) — AI Challenge

주변에서 흔히 볼 수 있는 **재활용품 사진**을 보고, 사진 속 재활용품의 **종류와 개수**를 묻는 4지선다 질문에
답하는 **Vision-Language 모델 파인튜닝** 프로젝트입니다.

---

## 1. 프로젝트 배경

SSAFY에서 진행된 AI Challenge 대회이며, **데이터 수집부터 모델링까지 참가자가 직접 수행**하는 형태였습니다.

| 단계 | 내용 |
|------|------|
| ① 데이터 수집 | 참가자 1인당 재활용품 사진 약 **20장** 촬영·제출 (쓰레기장, 실제 음료통 등 생활 속 이미지) |
| ② 라벨링 | 참가자 1인당 약 **100장**의 사진을 보며 "어떤 재활용품인지 / 몇 개인지" 라벨링 |
| ③ 모델링 | ①②로 구축된 데이터셋으로 예측 모델 개발 |
| ④ 평가 | 모델 종류 제한 없음. **테스트셋 정확도(Accuracy)가 가장 높은 팀이 우승** |

즉, 데이터셋 자체가 참가자들이 직접 만든 것이라 **촬영 환경·조명·구도·라벨 품질의 편차가 큰**,
현실적인 노이즈를 그대로 가진 데이터였습니다.

### 데이터 형태

라벨링 결과는 다음과 같은 4지선다 VQA 형식의 CSV로 제공되었습니다.

```
id, path, question, a, b, c, d, answer
```

- `path` : 이미지 상대 경로 (`train/xxx.jpg`, `test/xxx.jpg`)
- `question` : 예) "이 사진에 있는 재활용품의 종류는?", "페트병은 몇 개인가?"
- `a` ~ `d` : 보기
- `answer` : 정답 (`a` / `b` / `c` / `d`)

| 파일 | 규모 | 비고 |
|------|------|------|
| `train.csv` | 약 5,073건 | 라벨 확정 데이터 |
| `dev.csv` | 약 4,413건 | 어노테이터 5명의 답변 → **다수결(2표 이상)** 로 라벨 생성 |
| `test.csv` | — | 정답 비공개, 제출용 |

> ⚠️ 데이터셋(CSV·이미지)은 대회 자산이므로 이 저장소에 포함되어 있지 않습니다.

---

## 2. 접근 방식

객체 탐지(YOLO 등)로 "종류 + 개수"를 직접 세는 방식 대신,
**멀티모달 LLM(VLM)을 4지선다 문제 풀이기로 파인튜닝**하는 방향을 선택했습니다.

- 질문 유형이 종류/개수/재질 등으로 다양해 **탐지 클래스 정의가 어려웠음**
- 라벨이 이미 "보기 중 하나"로 정규화되어 있어 **분류 문제로 환원 가능**
- 사전학습된 VLM의 상식(투명도, 재질, 라벨 텍스트 판별)을 그대로 활용 가능

### 핵심 기법

| 기법 | 설명 |
|------|------|
| **QLoRA (4-bit NF4)** | 8B 모델을 단일 GPU(Colab A100/L4)에서 학습 가능하게 압축 |
| **LoRA r=16, α=32** | Attention/MLP 프로젝션에만 어댑터 삽입 → 학습 파라미터 최소화 |
| **레이블 마스킹** | 프롬프트 토큰은 `labels=-100`으로 마스킹하고 **정답 토큰에만 loss** 적용 |
| **Logit 직접 비교 추론** | 텍스트를 생성하지 않고 첫 토큰 위치에서 `a/b/c/d` 로짓만 비교 → **파싱 실패 0%, 약 2배 빠름** |
| **이미지 증강** | Flip / ColorJitter / Rotation / RandomAffine (촬영 편차 대응) |
| **Cosine LR + Warmup(5%)** | 짧은 학습에서의 수렴 안정화 |
| **Gradient Clipping** | `max_norm=1.0` |
| **dev 다수결 라벨 활용** | 어노테이터 5명 중 2표 이상 일치한 샘플을 추가 학습 데이터로 편입 |

#### 왜 Logit 비교 추론인가

```python
# 생성 방식: 모델이 토큰을 생성 → 문자열 파싱 → "정답은 (b)입니다" 같은 응답에서 실패 가능
# Logit 방식: assistant 턴 첫 토큰 분포에서 a/b/c/d 로짓만 argmax → 항상 4개 중 하나
last_logits   = outputs.logits[0, -1, :]
choice_logits = last_logits[choice_token_ids]   # (4,)
pred = CHOICES[choice_logits.argmax().item()]
```

프롬프트에도 "소문자 한 글자만 출력"을 명시했지만, 생성 방식은 여전히 형식을 이탈하는 경우가 있어
**로짓 비교로 전환한 뒤 파싱 오류가 완전히 사라졌습니다.**

---

## 3. 실행 방법

### 3.1 환경

- Python 3.10+, CUDA GPU (VRAM 16GB 이상 권장 — 8B + 4bit 기준)
- bfloat16 미지원 GPU(T4 등)는 자동으로 float16 폴백

```bash
pip install "transformers>=4.43.2,<5.0.0" "accelerate>=0.34.2" \
            "peft>=0.13.2" "bitsandbytes>=0.43.3" \
            einops timm sentencepiece tiktoken pandas pillow
```

### 3.2 로컬 실행 (`internvl2_8b_local.ipynb`)

작업 폴더를 아래 구조로 준비한 뒤 노트북을 위에서부터 실행합니다.

```
(작업 폴더)/
├── train.csv
├── test.csv
├── train/      ← 학습 이미지
└── test/       ← 테스트 이미지
```

경로가 다르면 노트북 상단의 `BASE_DIR`만 수정하면 됩니다.

### 3.3 Colab 실행 (`internvl2_8b_colab*.ipynb`)

Google Drive의 `MyDrive` 아래에 데이터를 두고 실행합니다.

```
MyDrive/
├── subset_train.csv / subset_val.csv / test.csv   (colab.ipynb)
├── train.csv / test.csv                            (colab_full_train.ipynb)
├── train/  dev/  test/
└── test.zip   ← 있으면 /content 로 자동 압축 해제
```

### 3.4 출력

학습이 끝나면 LoRA 어댑터가 `internvl2_vqa_lora/` (또는 `qwen_vqa_lora/`)에 저장되고,
테스트 추론 결과가 `submission.csv`로 생성됩니다.

```csv
id,answer
0,b
1,d
...
```

---

## 4. 주요 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| 모델 | `OpenGVLab/InternVL2-8B` (최종) / `Qwen/Qwen2.5-VL-3B-Instruct` (초기) |
| 이미지 크기 | 448×448 (InternVL 권장) / 384×384 (Qwen) |
| 정규화 | ImageNet mean·std |
| 양자화 | 4-bit NF4 + double quant |
| LoRA | r=16, α=32, dropout=0.05 |
| LoRA target | InternVL2: `wqkv, wo, w1, w2, w3` / Qwen: `q,k,v,o_proj, gate,up,down_proj` |
| Optimizer | AdamW (weight_decay=0.01) |
| LR | 1e-4 (InternVL2) / 2e-4 (Qwen) |
| Batch | 1 × grad accum 4 (유효 배치 4) |
| Epochs | 1~3 |
| Seed | 42 |

---

## 5. 베이스라인 대비 개선 내역

| 항목 | 베이스라인 | 본 솔루션 |
|------|-----------|-----------|
| 학습 데이터 | 200개 | 전체 train(5,073) + dev 다수결(4,413) |
| Epochs | 1 | 3 |
| LoRA rank | r=8 | r=16 |
| LR 스케줄 | linear | cosine + warmup |
| Gradient Clipping | 없음 | max_norm=1.0 |
| 레이블 마스킹 | 없음 (전체 토큰 loss) | 정답 토큰만 loss |
| 추론 방식 | 텍스트 생성 + 파싱 | **logit 직접 비교** |
| 이미지 증강 | 없음 | Flip / Jitter / Rotation / Affine |

---

## 6. 회고

**결과적으로 수상하지는 못했습니다.** 돌아보며 정리한 한계와 배운 점입니다.

- **컴퓨팅 자원이 병목이었다.** 8B 모델 + Colab 환경에서 에폭당 학습 시간이 길어,
  초반에는 학습 샘플을 200~2,300개로 줄여가며 실험할 수밖에 없었습니다.
  하이퍼파라미터 탐색 사이클을 충분히 돌리지 못한 것이 가장 큰 아쉬움입니다.
- **검증 전략이 늦게 잡혔다.** 초기 노트북은 val **loss** 기준으로 best를 저장했는데,
  대회 지표는 accuracy였습니다. 이후 `internvl2_8b_colab.ipynb`에서 val accuracy 기준으로 바꿨지만,
  최종 제출본은 시간 문제로 검증 없이 전체 데이터를 학습한 버전이었습니다.
- **개수 세기(counting) 문제에 약했다.** 종류·재질 분류는 사전학습 지식으로 잘 맞췄지만,
  "몇 개인가" 유형은 VLM이 구조적으로 취약했습니다. 탐지 모델을 보조로 앙상블했다면
  더 나았을 것으로 봅니다.
- **라벨 노이즈.** 참가자가 직접 만든 데이터라 애매한 라벨이 섞여 있었고,
  dev 다수결로 일부 완화했지만 train 쪽 노이즈는 별도로 정제하지 못했습니다.

**그럼에도 얻은 것:** QLoRA로 8B급 VLM을 단일 GPU에서 파인튜닝하는 전 과정,
레이블 마스킹으로 학습 신호를 정답 토큰에 집중시키는 방법,
그리고 **생성 대신 로짓을 직접 비교하는 객관식 추론 트릭**은 이후 작업에도 그대로 쓸 수 있는 자산이 되었습니다.

---

## 7. 참고

- [InternVL2](https://github.com/OpenGVLab/InternVL) — OpenGVLab
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Alibaba
- [PEFT (LoRA)](https://github.com/huggingface/peft) — Hugging Face
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)

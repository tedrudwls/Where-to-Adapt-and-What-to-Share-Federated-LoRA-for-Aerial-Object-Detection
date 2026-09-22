# AOD-4 RT-DETR + LoRA 연합학습 실험 재현 가이드

이 문서는 다음 비교를 동일한 데이터와 학습 budget으로 재현하기 위한 실행 문서다.

- 단독학습: `solo/full_ft`, `solo/lora`
- 중앙집중형: `centralized/full_ft`, `centralized/lora`
- 연합학습: `fl/full_ft`, `fl/lora`, `fl/fedsa_lora`,
  `fl/fixed_share_b_lora`(고정 factor-role 대조군)
- 이질성: label-blind random IID baseline과 Dirichlet `alpha={0.1, 0.4, 0.5, 1.0}`
- LoRA rank: `r={4, 8, 16}`, 항상 `lora_alpha=2r`
- LoRA target: decoder-only, backbone-only, both
- image-level loss Membership Inference Attack(MIA)

Primary setting은 client 3개, paired partition/training seed
`{(42,42),(43,43),(44,44)}`, RT-DETR-L, 640 입력,
FL `20 rounds x 5 local epochs`,
단독/중앙집중형 `100 epochs`이다. 모든 client가 매 round 참여하면
명목상 data pass는 FL과 중앙집중형 모두 `100 x 전체 train image 수`가
된다. 단, optimizer state와 client별 mini-batch 구성이 다르므로 두 학습의
optimization trajectory가 같다는 뜻은 아니다.

## 1. 서버 경로

기본 스크립트는 다음 경로를 사용한다.

```text
프로젝트  /home/gpuadmin/kim/fedsalora
데이터    /home/gpuadmin/kim/project2/data/aod4/AOD4/Images
train     .../train/_annotations.coco.json
val       .../val/_annotations.coco.json
test      .../test/_annotations.coco.json
결과      /home/gpuadmin/kim/fedsalora/results/official_v6/seed_<SEED>/
로그      /home/gpuadmin/kim/fedsalora/logs/official_v6/seed_<SEED>/
split     /home/gpuadmin/kim/fedsalora/data/splits/
```

다른 위치에서 실행할 때는 파일을 수정하지 말고 환경변수로 덮어쓴다.

```bash
unset PROJECT_DIR DATA_ROOT SPLIT_DIR RESULTS_ROOT LOG_ROOT MODEL_WEIGHTS PYTHON_BIN
export PROJECT_DIR=/home/gpuadmin/kim/fedsalora
export DATA_ROOT=/home/gpuadmin/kim/project2/data/aod4/AOD4/Images
export SPLIT_DIR=/home/gpuadmin/kim/fedsalora/data/splits
export RESULTS_ROOT=/home/gpuadmin/kim/fedsalora/results/official_v6
export LOG_ROOT=/home/gpuadmin/kim/fedsalora/logs/official_v6
# 권장: 논문용으로 고정해 둔 local checkpoint의 절대경로
# export MODEL_WEIGHTS=/home/gpuadmin/kim/fedsalora/rtdetr-l.pt
cd "$PROJECT_DIR"
```

첫 `unset`은 이전 shell session에 남아 있을 수 있는 예전 프로젝트·split·
결과·weight·Python 경로가 새 기본값보다 우선하는 것을 방지한다.
고정 checkpoint나 별도 Python executable을 사용하면 이 블록 뒤에서
새 절대경로로 다시 export한다.
이 문서의 이후 모든 상대경로 명령은 `cd "$PROJECT_DIR"`가 실행된
같은 shell을 전제로 한다. 임의의 command block만 별도로 복사해 실행할
때도 먼저 `cd /home/gpuadmin/kim/fedsalora`를 실행한다.

`MODEL_WEIGHTS`는 선택 사항이며 비워 두면 `rtdetr-l.pt`를 model name으로
해석하는 기존 Ultralytics 동작을 사용한다. 이 기본 경로를 쓴 기존
결과를 재사용할 때는 `${PROJECT_DIR}/rtdetr-l.pt`가 계속 존재하고 기록된
SHA-256과 일치해야 한다. 논문용 실행에서는 자동 다운로드에 의존하지
말고 immutable local checkpoint의 절대경로를 지정한 뒤 그 파일도
artifact와 함께 보존하는 편이 안전하다. 결과에는 실제 checkpoint file SHA-256과
pretrained tensor-state SHA-256이 기록된다.

Ultralytics의 공식 `rtdetr-l.pt` asset은 버전에 따라 내부 checkpoint
container가 `DetectionModel`로 직렬화되어 있을 수 있다. 이것만으로 YOLO
모델로 판단하지 않는다. 이 코드는 실제 마지막 module이
`RTDETRDecoder`인지, RT-DETR YAML의 backbone/head가 존재하는지를 모두
검증한 다음 학습용 모델을 `RTDETRDetectionModel`로 다시 구성한다. 반대로
일반 YOLO `DetectionModel`은 마지막 head가 `RTDETRDecoder`가 아니므로
즉시 거부된다. 결과 architecture에는 원본 container class, head class와
이 compatibility representation을 함께 기록한다. 표준 Ultralytics trainer가
설정하는 outer `model.nc`도 수동 FL loop에서 loss를 직접 호출하기 전에
YAML/head의 4 classes와 같은 값으로 명시해 둔다.

Ultralytics `val()`과 `predict()`는 내부적으로 inference mode에서 backend를
준비한다. FL client처럼 학습 직후 CPU로 offload된 모델은 이 내부에서 처음
CUDA로 이동시키면 trainable parameter가 inference tensor로 바뀔 수 있다.
따라서 이 artifact는 평가 함수에 들어가기 전에 동일 model 객체를 요청
device의 FP32로 이동시키고, 평가 후 원 device/dtype, `requires_grad`,
train/eval mode와 RT-DETR decoder anchor cache를 복원한다. 평가 정밀도는
`quantize=None`으로 명시하여 FP32로 고정한다. 이 순서는 optimizer가 참조하는
Parameter 객체가 교체되지 않았는지도 검사한다.

## 2. 환경 설치

현재 서버에서 검증한 interpreter는 Python 3.9.18이다. 모든 union annotation
파일은 `from __future__ import annotations`를 사용하므로 Python 3.9에서
동작한다. 재현 환경은 Python 3.9–3.11 범위의 virtual environment 사용을
권장한다. `models/rtdetr_lora.py`가 RT-DETR
내부 module 구조를 사용하므로 `ultralytics==8.4.126`은 반드시 유지해야 한다.
`main.py`가 canonical entry point이며, 기존 서버 명령과의 호환을 위한
`federated_main.py`도 같은 CLI를 그대로 전달하는 wrapper로 남아 있다.

```bash
cd /home/gpuadmin/kim/fedsalora
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

이전 프로젝트의 `.venv`를 새 루트로 복사했다면 그대로 사용하지
않는다. virtualenv launcher의 shebang과 내부 경로에 이전 절대경로가
남을 수 있다. 기존 `.venv`를 별도 위치에 보관한 뒤 위 명령으로
`/home/gpuadmin/kim/fedsalora/.venv`를 새로 생성한다.

서버 CUDA driver와 `requirements.txt`의 PyTorch wheel이 맞지 않으면 PyTorch
공식 설치 명령으로 `torch==2.5.1`, `torchvision==0.20.1`의 해당 CUDA build를
먼저 설치한 뒤 나머지 requirements를 설치한다. 실행 환경은 논문 artifact에
반드시 함께 보존한다.

```bash
python3 - <<'PY'
import torch, torchvision, ultralytics
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert ultralytics.__version__ == "8.4.126"
PY
python3 -m pip freeze > environment.freeze.txt
```

## 3. 데이터 검증과 immutable split 준비

`prepare_split.py`는 원본 COCO annotation을 검증하고 다음을 수행한다.

Primary 실험은 Roboflow AOD-4 v6가 제공한 train/val/test membership을
그대로 보존한다. 서버 원본에서 확인한 공식 수치는 다음과 같으며 전처리 후에도
이미지와 annotation 수가 모두 같아야 한다.

| split | images | annotations | background images |
|---|---:|---:|---:|
| train | 15,761 | 22,058 | 415 |
| val | 4,514 | 6,369 | 125 |
| test | 2,241 | 3,171 | 56 |
| total | 22,516 | 31,598 | 596 |

원본에는 `.rf.<hash>`를 제거한 Roboflow source key 기준 공식 split 간
overlap이 발견됐지만, Primary에서는 이를 기록만 하고 이미지를 제외하거나
다른 split으로 이동하지 않는다. manifest의
`cross_split_source_audit`와 `post_policy_cross_split_source_check`에 overlap
수치가 남는다. 따라서 결과는 **official AOD-4 v6 split benchmark**로
기술하되, source/video-disjoint 일반화 실험이라고 표현하면 안 된다.

- Roboflow source key와 exact-byte SHA-256을 이용한 source-component 진단은 유지
- 공식 train/val/test membership 보존 및 excluded image 0개 검증
- pinned v6 count gate로 22,516 images, 31,598 annotations, 596 background
  images 일치 검증
- 각 공식 split 내부에서는 source component를 쪼개지 않고, 공식 split 사이에
  같은 component가 있으면 동일 client ID에 귀속시키는 source-group-atomic
  client partition 적용
- Dirichlet은 group별 multi-object class histogram을 고려한 greedy partition 후
  whole-group balance repair를 적용하여 client image 수의 `max-min`이
  가장 큰 분할 불가 source group 크기보다 크지 않게 제한
- IID는 label을 보지 않는 seeded random source-group split without replacement로,
  동일한 indivisible-group 수량 균형 bound를 적용
- annotation이 없는 background image 유지
- COCO crowd-ignore 의미를 YOLO label이 보존할 수 없으므로 `iscrowd=1`이 한 건이라도
  있으면 fail-fast(AOD-4 primary는 crowd annotation 0건을 요구)
- 모든 image에 Pillow 구조 `verify()`와 전체 pixel `load()`를 모두
  수행하여 truncated image를 차단하고 COCO width/height 일치 검사
- 공식 train/val/test 각각을 client별로 분리
- COCO category ID를 contiguous YOLO label로 변환
- raw annotation SHA-256, 모든 raw image의 filename/content SHA-256 inventory,
  source-component overlap audit, 공식 split 통계를 schema-v7 JSON manifest에
  저장
- Roboflow export가 선언한 `airplane-helicopter-drone-bird` 같은 비표적
  메타데이터 category는 raw annotation 참조가 정확히 0건일 때만 4개 표적
  클래스에서 제외하고, split별 선언 category·raw count·제외 내역을 manifest에
  기록한다. 비표적 category가 한 건이라도 GT annotation에서 사용되면 자동
  삭제·병합하지 않고 fail-fast한다.
- 생성된 client/full YAML, label byte와 image symlink target 전체의 SHA-256 저장
  (`*.cache`만 제외) 및 모든 학습 시작 때 재검증
- 공식 split별 client assignment가 중복 없는 정확한 cover인지, 각 split 내부와
  train/val/test 사이에서 같은 source group의 client-owner 충돌이 0개인지 재검증
- source hash inventory 생성과 재현은 필수이며 논문용 split에서
  비활성화할 수 없음

기존 source-exclusive split은 삭제하거나 이동하지 않는다. 특히 기존
`yolo_dirichlet_.../dataset.yaml`에는 생성 당시 절대경로가 들어 있고 manifest의
YOLO-tree digest도 그 경로를 검증하므로, 폴더를 옮기면 기존 artifact의 재현성이
깨질 수 있다. 새 공식 artifact는 다음처럼 `official_v6` prefix를 사용하므로 같은
`data/splits` 안에서 안전하게 공존한다.

```text
기존 정제 pilot: split_dirichlet_a0.4_c3_s42.json
                 yolo_dirichlet_a0.4_c3_s42/
새 공식 Primary: split_official_v6_dirichlet_a0.4_c3_s42.json
                 yolo_official_v6_dirichlet_a0.4_c3_s42/
```

기존 checkpoint/result도 수정하거나 resume하지 않고 pilot artifact로 보존한다.
새 launcher는 기본적으로 `results/official_v6`와 `logs/official_v6`를 사용하므로
최종 집계에 기존 결과가 섞이지 않는다. 기존 정제 split이 필요해도 `--overwrite`를
사용하거나 YAML/JSON의 경로를 수동 치환하지 않는다.

학습 시작 때 annotation과 source image tree가 manifest와 같은지도 다시
확인한다. 같은 manifest로 반복 실행할 때는 최초의 전체 content hash 검증 뒤
파일명·크기·mtime·ctime fingerprint가 모두 같을 때만 검증 cache를 사용한다.
스토리지 snapshot 복원처럼 stat이 신뢰되지 않는 상황에서는 모든 학습 명령에
`--rehash_source_images`를 추가한다. `scripts/preflight.py`는 cache와 무관하게
모든 raw image content hash를 다시 계산하고, 공식 membership 보존·source
overlap audit·client group assignment를 manifest와 독립적으로 재현한다.

기본 Dirichlet alpha=0.4, seed=42:

```bash
cd /home/gpuadmin/kim/fedsalora
python3 scripts/prepare_split.py \
  --data_root /home/gpuadmin/kim/project2/data/aod4/AOD4/Images \
  --output_dir /home/gpuadmin/kim/fedsalora/data/splits \
  --source_split_policy official_aod4_v6 \
  --partition dirichlet --alpha 0.4 --num_clients 3 --seed 42
```

생성 파일은 다음과 같다.

```text
data/splits/split_official_v6_dirichlet_a0.4_c3_s42.json
data/splits/yolo_official_v6_dirichlet_a0.4_c3_s42/client_0/dataset.yaml
data/splits/yolo_official_v6_dirichlet_a0.4_c3_s42/client_1/dataset.yaml
data/splits/yolo_official_v6_dirichlet_a0.4_c3_s42/client_2/dataset.yaml
data/splits/yolo_official_v6_dirichlet_a0.4_c3_s42/full/dataset.yaml
```

Publication suite에서 사용할 partition seed 42/43/44의 모든 alpha와
명시적 IID split을 미리 만들려면 다음을 실행한다. 각 paired
replicate 내에서는 모든 방법이 동일 manifest를 공유하고, replicate 간에는
split realization과 training randomness를 함께 바꿔 이질적 분할에 대한
결과의 안정성을 평가한다.

```bash
cd /home/gpuadmin/kim/fedsalora
for partition_seed in 42 43 44; do
  for alpha in 0.1 0.4 0.5 1.0; do
    python3 scripts/prepare_split.py \
      --data_root /home/gpuadmin/kim/project2/data/aod4/AOD4/Images \
      --output_dir /home/gpuadmin/kim/fedsalora/data/splits \
      --source_split_policy official_aod4_v6 \
      --partition dirichlet --alpha "$alpha" --num_clients 3 --seed "$partition_seed"
  done
  python3 scripts/prepare_split.py \
    --data_root /home/gpuadmin/kim/project2/data/aod4/AOD4/Images \
    --output_dir /home/gpuadmin/kim/fedsalora/data/splits \
    --source_split_policy official_aod4_v6 \
    --partition iid --num_clients 3 --seed "$partition_seed"
done
```

IID는 큰 alpha로 근사하지 않고 `--partition iid`로 별도 생성한다.
이 mode는 label을 사용하지 않고 source group을 크기 내림차순 LPT로
현재 image 수가 가장 작은 client에 복원 없이 배정하며, 동일 group 크기와
동일 client load의 tie만 seed로 해소한다. group 크기가 1보다 클 수 있으므로 client image 수의
`max-min` 차이가 1을 넘을 수 있으며, 이는 source 누수 방지를 위한
의도된 trade-off다. 파일명도 `split_official_v6_iid_c3_s42.json`처럼
Dirichlet manifest와 분리된다. 다만 finite multi-object sample의 한 실현이므로
client의 realized class distribution이 정확히 같을 필요는 없다. manifest의
class histogram과 divergence를 함께 보고해야 한다.

긴 학습 전에는 unit test, canonical preprocessing audit, 실제 model injection과
labeled forward/backward 검사를 순서대로 실행한다.

```bash
python3 -m unittest \
  tests.test_dataset tests.test_lora tests.test_factor_sharing \
  tests.test_aggregate_results tests.test_summarize_unseen_class_case
python3 scripts/check_dataset.py \
  --data_root /home/gpuadmin/kim/project2/data/aod4/AOD4/Images \
  --source_split_policy official_aod4_v6 \
  --partition dirichlet --dirichlet_alpha 0.4 --num_clients 3 --seed 42
```

```bash
for factor_method in fedsa_lora fixed_share_b_lora; do
  python3 scripts/preflight.py \
    --data_root /home/gpuadmin/kim/project2/data/aod4/AOD4/Images \
    --split_file /home/gpuadmin/kim/fedsalora/data/splits/split_official_v6_dirichlet_a0.4_c3_s42.json \
    --model_weights /home/gpuadmin/kim/fedsalora/rtdetr-l.pt \
    --partition dirichlet --dirichlet_alpha 0.4 \
    --partition_seed 42 --seed 42 \
    --mode fl --fl_method "$factor_method" \
    --lora_rank 8 --lora_alpha 16 --device cuda
done
```

`preflight.py`는 실제 RT-DETR loss의 1개 labeled batch forward/backward까지
수행하지만 optimizer step이나 장기 학습·test 평가는 하지 않는다.

optimizer step, aggregation, checkpoint, evaluation, detection visualization까지 짧게 확인하는 1-round
smoke run은 다음과 같다. 이 결과는 논문 집계에 넣지 않는다.

```bash
python3 main.py \
  --data_root /home/gpuadmin/kim/project2/data/aod4/AOD4/Images \
  --split_file /home/gpuadmin/kim/fedsalora/data/splits/split_official_v6_dirichlet_a0.4_c3_s42.json \
  --model_weights /home/gpuadmin/kim/fedsalora/rtdetr-l.pt \
  --output_dir /home/gpuadmin/kim/fedsalora/results_smoke \
  --log_dir /home/gpuadmin/kim/fedsalora/logs_smoke \
  --exp_name fedsa_1round --partition dirichlet --dirichlet_alpha 0.4 \
  --partition_seed 42 --seed 42 --mode fl --fl_method fedsa_lora \
  --fl_rounds 1 --local_epochs 1 --lora_rank 8 --lora_alpha 16 \
  --device cuda --no-amp --close_mosaic_epochs 0 \
  --visualize_interval 1 --vis_samples 2 --no-cross_client_eval
```

smoke run이 완료 checkpoint를 쓰기 전에 실패했다면 `--resume`하거나 같은
실험명을 재사용하지 않는다. 기존 디렉터리는 진단용 pilot artifact로 보존하고
`--exp_name fedsa_1round_retry1`처럼 새 이름으로 pretrained checkpoint부터
다시 실행한다. split tree와 `labels.cache`는 그대로 재사용해도 된다.

## 4. Primary method별 직접 실행 명령

아래 명령은 training seed=42, partition seed=42, Dirichlet alpha=0.4인
하나의 paired replicate 예시다. publication 반복 43/44에서는 각각
`..._s43.json`/`..._s44.json`과 `--partition_seed 43`/`44`를 사용하고
training seed도 같이 바꾼다. 서로 다른 방법은 각 replicate에서 반드시
동일한 manifest를 사용한다.

공통 shell 변수:

```bash
cd /home/gpuadmin/kim/fedsalora
DATA=/home/gpuadmin/kim/project2/data/aod4/AOD4/Images
SPLIT=/home/gpuadmin/kim/fedsalora/data/splits/split_official_v6_dirichlet_a0.4_c3_s42.json
OUT=/home/gpuadmin/kim/fedsalora/results/official_v6/seed_42
LOG=/home/gpuadmin/kim/fedsalora/logs/official_v6/seed_42
```

단독학습 full fine-tuning, client 0:

```bash
python3 main.py --data_root "$DATA" --split_file "$SPLIT" \
  --output_dir "$OUT" --log_dir "$LOG" --exp_name solo_full_ft_client0 \
  --partition dirichlet --dirichlet_alpha 0.4 --partition_seed 42 --seed 42 \
  --mode solo --client_id 0 --fl_method full_ft --solo_epochs 100 \
  --batch_size 8 --img_size 640 --model_name rtdetr-l --patience 0 --no-amp
```

단독학습 LoRA, client 0:

```bash
python3 main.py --data_root "$DATA" --split_file "$SPLIT" \
  --output_dir "$OUT" --log_dir "$LOG" --exp_name solo_lora_client0 \
  --partition dirichlet --dirichlet_alpha 0.4 --partition_seed 42 --seed 42 \
  --mode solo --client_id 0 --fl_method lora --solo_epochs 100 \
  --lora_rank 8 --lora_alpha 16 \
  --batch_size 8 --img_size 640 --model_name rtdetr-l --patience 0 --no-amp
```

client 1과 2는 `--client_id`와 `--exp_name`을 각각 바꾼다.

중앙집중형 full fine-tuning:

```bash
python3 main.py --data_root "$DATA" --split_file "$SPLIT" \
  --output_dir "$OUT" --log_dir "$LOG" --exp_name centralized_full_ft \
  --partition dirichlet --dirichlet_alpha 0.4 --partition_seed 42 --seed 42 \
  --mode centralized --fl_method full_ft --centralized_epochs 100 \
  --batch_size 8 --img_size 640 --model_name rtdetr-l --patience 0 --no-amp
```

중앙집중형 LoRA:

```bash
python3 main.py --data_root "$DATA" --split_file "$SPLIT" \
  --output_dir "$OUT" --log_dir "$LOG" --exp_name centralized_lora \
  --partition dirichlet --dirichlet_alpha 0.4 --partition_seed 42 --seed 42 \
  --mode centralized --fl_method lora --centralized_epochs 100 \
  --lora_rank 8 --lora_alpha 16 \
  --batch_size 8 --img_size 640 --model_name rtdetr-l --patience 0 --no-amp
```

FL + full fine-tuning:

```bash
python3 main.py --data_root "$DATA" --split_file "$SPLIT" \
  --output_dir "$OUT" --log_dir "$LOG" --exp_name fl_full_ft_a0.4 \
  --partition dirichlet --dirichlet_alpha 0.4 --partition_seed 42 --seed 42 \
  --mode fl --fl_method full_ft --fl_rounds 20 --local_epochs 5 \
  --batch_size 8 --img_size 640 --model_name rtdetr-l --patience 0 \
  --no-amp --reset_optimizer_each_round
```

FL + LoRA(A와 B 모두 집계):

```bash
python3 main.py --data_root "$DATA" --split_file "$SPLIT" \
  --output_dir "$OUT" --log_dir "$LOG" --exp_name fl_lora_r8_a0.4 \
  --partition dirichlet --dirichlet_alpha 0.4 --partition_seed 42 --seed 42 \
  --mode fl --fl_method lora --fl_rounds 20 --local_epochs 5 \
  --lora_rank 8 --lora_alpha 16 \
  --batch_size 8 --img_size 640 --model_name rtdetr-l --patience 0 \
  --no-amp --reset_optimizer_each_round
```

FL + FedSA-LoRA(global A와 task head, local B):

```bash
python3 main.py --data_root "$DATA" --split_file "$SPLIT" \
  --output_dir "$OUT" --log_dir "$LOG" --exp_name fl_fedsa_lora_r8_a0.4 \
  --partition dirichlet --dirichlet_alpha 0.4 --partition_seed 42 --seed 42 \
  --mode fl --fl_method fedsa_lora --fl_rounds 20 --local_epochs 5 \
  --lora_rank 8 --lora_alpha 16 \
  --batch_size 8 --img_size 640 --model_name rtdetr-l --patience 0 \
  --no-amp --reset_optimizer_each_round
```

FL + Fixed Share-B LoRA(global B와 task head, local A):

```bash
python3 main.py --data_root "$DATA" --split_file "$SPLIT" \
  --output_dir "$OUT" --log_dir "$LOG" \
  --exp_name fl_fixed_share_b_lora_r8_a0.4 \
  --partition dirichlet --dirichlet_alpha 0.4 --partition_seed 42 --seed 42 \
  --mode fl --fl_method fixed_share_b_lora --fl_rounds 20 --local_epochs 5 \
  --lora_rank 8 --lora_alpha 16 \
  --batch_size 8 --img_size 640 --model_name rtdetr-l --patience 0 \
  --no-amp --reset_optimizer_each_round
```

여기서 `Fixed`는 B를 freeze한다는 뜻이 아니라, **B를 공유하는 역할 배치를
학습 내내 고정**한다는 뜻이다. A와 B는 둘 다 각 client의 local optimizer로
학습되며, B+task head만 통신·FedAvg되고 A는 client-local로 유지된다.

기본 optimizer는 AdamW다. full FT의 base LR은 `1e-4`, full-FT
backbone은 그 `0.1` 배, LoRA/FedSA/Fixed Share-B adapter는 `3e-4`, global task head는
`1e-4`이다. weight decay는 `1e-4`, warmup은 effective 5 epochs, cosine
minimum LR ratio는 `0.01`, gradient clip은 `0.1`이다. Primary에서 AMP는
안정성을 위해 기본 `false`로 고정한다. 결과에서는 `--patience 0`으로
early stopping을 끄고 validation AP로 checkpoint를 선택한다. test는 학습,
stopping 또는 checkpoint 선택에 사용하면 안 된다.
단독학습은 해당 client validation AP, 중앙집중형은 세 client-local validation
AP의 macro 평균, FL도 세 client-local validation AP의 macro 평균으로 best
checkpoint를 선택한다. 따라서 중앙집중형만 pooled validation으로 선택되는
불공정한 차이는 두지 않았다. 선택된 checkpoint에서 test는 마지막에 한 번만
평가한다.
FL은 매 server broadcast 후 local AdamW moment를 reset하지만 warmup-cosine
step은 100 effective local epochs 전체에 걸쳐 연속적으로 진행한다.
입력 resize/letterbox와 train augmentation은 고정한 Ultralytics RT-DETRDataset
구현을 모든 방법에서 공유한다. 초기 확률은 HSV `(0.015,0.7,0.4)`,
translate `0.1`, scale `0.5`, horizontal flip `0.5`, mosaic `1.0`이며 rotation,
shear, perspective, vertical flip, mixup, cutmix, copy-paste는 `0`이다. 공식
Trainer와 같은 방식으로 effective epoch 91 시작 전에 mosaic 계열 augmentation을
닫아 마지막 10 epochs에는 적용하지 않는다(`--close_mosaic_epochs 10`). 실제
값과 전환 규칙은 각 결과의 `augmentation_protocol`에도 저장된다.

## 5. 자동 실행 스크립트

아래 suite 스크립트는 기본적으로 seed 42/43/44를 실행한다. 기존 JSON은 단순히
파일이 있다는 이유로 건너뛰지 않는다. complete/schema marker와 mode, method,
training/partition seed, model 및 선택적 weight-file SHA-256, batch, active budget,
rank/scaling, LoRA target, client/worker 수, optimizer/LR schedule, AMP/FedProx,
augmentation/close-mosaic, visualization 설정, 실제 split-file SHA-256이 현재
명령과 모두 일치할 때만 재사용한다. 또한 completed run/evaluation manifest,
mode별 checkpoint, 필수 plot과 예정된 detection visualization artifact를 검증한다.
불일치하면 fail-fast하며, 의도적인
재계산은 별도 `RESULTS_ROOT` 또는 `FORCE_RERUN=1`을 사용한다. `run_mia.sh`는
MIA sample cap과 calibration fraction까지 비교하고, 이 공격 설정만 다르면 기존
학습 checkpoint를 유지한 채 MIA evaluation만 다시 수행한다.

전체 suite:

```bash
cd /home/gpuadmin/kim/fedsalora
bash scripts/run_all_experiments.sh
```

한 seed만 먼저 primary 전체 run:

```bash
SEEDS=42 RUN_IID=0 RUN_ALPHA=0 RUN_RANK=0 RUN_TARGET_ABLATION=0 \
  RUN_MIA=0 RUN_AGGREGATE=0 \
  bash scripts/run_all_experiments.sh
```

개별 suite:

```bash
bash scripts/run_iid_sensitivity.sh
bash scripts/run_alpha_sensitivity.sh
bash scripts/run_rank_sensitivity.sh
bash scripts/run_target_ablation.sh
bash scripts/run_factor_sharing_ablation.sh
bash scripts/run_mia.sh
```

간단한 1회 실행용 legacy launcher도 동일한 새 CLI와 서버 경로로
정리되어 있다.

```bash
# client 0,1,2를 각각 단독 LoRA 학습
bash scripts/run_solo.sh all lora

# 중앙집중형 full FT와 LoRA 둘 다
bash scripts/run_centralized.sh all

# FedSA-LoRA, Dirichlet alpha=0.4, rank=8
bash scripts/run_fl.sh fedsa_lora 0.4 8

# Fixed Share-B 대조군, 완전히 같은 split/seed/rank
bash scripts/run_fl.sh fixed_share_b_lora 0.4 8

# publication replicate 43: split과 training seed를 같이 변경
TRAIN_SEED=43 PARTITION_SEED=43 bash scripts/run_fl.sh fedsa_lora 0.4 8
```

각 스크립트의 범위:

| 스크립트 | 기본 실행 |
|---|---|
| `run_iid_sensitivity.sh` | label-blind random IID baseline에서 FL full FT, LoRA, FedSA-LoRA, Fixed Share-B |
| `run_alpha_sensitivity.sh` | FL full FT, LoRA, FedSA-LoRA, Fixed Share-B 각각 alpha 0.1/0.4/0.5/1.0 |
| `run_heterogeneity_cell.sh` | alpha=0.1 또는 IID의 method/seed 한 셀만 exact protocol로 실행 |
| `run_heterogeneity_queue.sh` | 한 condition/seed의 네 방법을 순차 실행하는 tmux queue |
| `launch_heterogeneity_24_tmux.sh` | 세 GPU에 여섯 queue를 충돌 검사 후 배치해 정확히 24개 실행 |
| `verify_heterogeneity_24.sh` | alpha=0.1+IID 24개 결과를 재학습 없이 exact gate로 검증 |
| `run_rank_sensitivity.sh` | FedSA-LoRA, rank 4/8/16, `lora_alpha=2r` |
| `run_target_ablation.sh` | FedSA-LoRA, decoder-only/backbone-only/both |
| `run_factor_sharing_ablation.sh` | primary alpha=0.4/rank=8에서 global-A/local-B와 global-B/local-A의 paired 비교 |
| `run_mia.sh` | Solo FT/LoRA, Central FT/LoRA, FL 네 방법의 loss-MIA |

Fixed Share-B는 primary factor-role 대조군이지만, 이질성에 따른 공유 방향의
민감도를 공정하게 비교하기 위해 IID/alpha sensitivity에는 포함한다. Rank와
target sensitivity는 각 전용 runner의 명시적 범위를 따른다.

환경변수 예시:

```bash
# publication 기본: replicate별로 split과 training randomness를 함께 변경
SEEDS="42 43 44" bash scripts/run_rank_sensitivity.sh

# 선택적 고정-split ablation: s42에서 training randomness만 변경
FIXED_PARTITION_SEED=42 SEEDS="42 43 44" \
  RESULTS_ROOT=/home/gpuadmin/kim/fedsalora/results_fixed_split \
  bash scripts/run_rank_sensitivity.sh

# alpha마다 네 FL 방법을 모두 비교
METHODS="full_ft lora fedsa_lora fixed_share_b_lora" \
  bash scripts/run_alpha_sensitivity.sh

# 이미 존재하는 JSON도 다시 계산. 기존 artifact가 섞이지 않도록 별도
# RESULTS_ROOT를 지정하는 편이 더 안전하다.
RESULTS_ROOT=/home/gpuadmin/kim/fedsalora/results_rerun \
  FORCE_RERUN=1 SEEDS=42 bash scripts/run_alpha_sensitivity.sh
```

기본 seed 정책은 `FIXED_PARTITION_SEED=match`이다. 즉 seed 42/43/44의
각 paired replicate에서 모든 방법은 동일 manifest를 공유하지만,
replicate 간에는 split realization과 training randomness를 모두 바꾼다.
`FIXED_PARTITION_SEED=42`는 동일 s42 split에서 학습 randomness만 변경하는
단일-분할 고정 ablation으로 유지한다. 두 seed 정책의 결과는 서로
다른 `RESULTS_ROOT`에 보존해야 한다.

### 5.1 Alpha=0.1 + explicit IID 24-run batch

다음 이질성 실험은 `2 conditions x 4 FL methods x 3 paired seeds = 24 runs`다.
각 replicate는 `(training seed, partition seed)=(42,42),(43,43),(44,44)`로
묶고, 같은 condition/seed 안의 네 방법은 동일 immutable split을 사용한다.
`alpha=0.1`은 강한 label skew이고 IID는 큰 alpha 근사가 아닌 label-blind
random source-group partition이다. 먼저 여섯 split을 직렬 생성하여 병렬 작업의
동시 cache/build 경합을 피한다.

```bash
cd /home/gpuadmin/kim/fedsalora
source .venv/bin/activate

bash scripts/prepare_heterogeneity_splits.sh
```

GPU마다 alpha queue와 IID queue를 하나씩 배치한다. 각 queue는 네 방법을
순차 실행하므로 GPU당 동시에 최대 두 학습 process만 존재한다. alpha queue는
Full FT부터, IID queue는 Fixed Share-B부터 시작하여 동일 GPU에서 Full FT 두 개가
동시에 시작되지 않게 했다. queue wrapper는 서버의 project/data/result/model 경로와
`device=cuda`, `CHECK_ONLY=0`을 다시 고정하므로 이전 tmux server 환경변수를
상속하지 않는다. 각 run 종료 직후 exact result gate가 자동 실행된다. 아래 여섯
session 이름이 이미 존재하면 먼저 기존 session 상태를 확인하고 새 실행을 시작하지
않는다.

여섯 session을 한 번에 안전하게 시작하려면 아래 launcher를 사용한다. 이 스크립트는
기존 session 하나라도 있거나 여섯 manifest/YOLO 쌍 중 하나라도 없으면 아무 queue도
시작하지 않고 종료한다.

```bash
bash scripts/launch_heterogeneity_24_tmux.sh
```

아래는 같은 배치를 수동으로 확인·실행할 때의 명령이다.

```bash
for session in \
  het_a01_s42 het_iid_s42 het_a01_s43 het_iid_s43 het_a01_s44 het_iid_s44; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[ERROR] Existing tmux session: $session"
    existing_sessions=1
  fi
done
if [[ "${existing_sessions:-0}" == "1" ]]; then
  echo "[STOP] Inspect or remove only the stale sessions above before launching."
  return 1 2>/dev/null || exit 1
fi
```

```bash
tmux new-session -d -s het_a01_s42 \
  "cd /home/gpuadmin/kim/fedsalora && source .venv/bin/activate && \
   export CUDA_VISIBLE_DEVICES=0 && \
   bash scripts/run_heterogeneity_queue.sh alpha0.1 42"
tmux set-window-option -t het_a01_s42:0 remain-on-exit on

tmux new-session -d -s het_iid_s42 \
  "cd /home/gpuadmin/kim/fedsalora && source .venv/bin/activate && \
   export CUDA_VISIBLE_DEVICES=0 && \
   bash scripts/run_heterogeneity_queue.sh iid 42"
tmux set-window-option -t het_iid_s42:0 remain-on-exit on

tmux new-session -d -s het_a01_s43 \
  "cd /home/gpuadmin/kim/fedsalora && source .venv/bin/activate && \
   export CUDA_VISIBLE_DEVICES=1 && \
   bash scripts/run_heterogeneity_queue.sh alpha0.1 43"
tmux set-window-option -t het_a01_s43:0 remain-on-exit on

tmux new-session -d -s het_iid_s43 \
  "cd /home/gpuadmin/kim/fedsalora && source .venv/bin/activate && \
   export CUDA_VISIBLE_DEVICES=1 && \
   bash scripts/run_heterogeneity_queue.sh iid 43"
tmux set-window-option -t het_iid_s43:0 remain-on-exit on

tmux new-session -d -s het_a01_s44 \
  "cd /home/gpuadmin/kim/fedsalora && source .venv/bin/activate && \
   export CUDA_VISIBLE_DEVICES=2 && \
   bash scripts/run_heterogeneity_queue.sh alpha0.1 44"
tmux set-window-option -t het_a01_s44:0 remain-on-exit on

tmux new-session -d -s het_iid_s44 \
  "cd /home/gpuadmin/kim/fedsalora && source .venv/bin/activate && \
   export CUDA_VISIBLE_DEVICES=2 && \
   bash scripts/run_heterogeneity_queue.sh iid 44"
tmux set-window-option -t het_iid_s44:0 remain-on-exit on
```

상태와 종료 코드를 확인한다.

```bash
tmux list-panes -a \
  -F 'session=#{session_name} dead=#{pane_dead} exit=#{pane_dead_status}' \
  | grep '^session=het_'
```

모두 끝난 뒤 다음 read-only gate는 누락·불완전 결과·protocol drift·split digest
불일치 중 하나라도 있으면 exit 1로 실패하며 새 학습을 시작하지 않는다.

```bash
bash scripts/verify_heterogeneity_24.sh
```

주의: `run_alpha_sensitivity.sh`를 직접 호출할 때 `ALPHAS="0.1"`을 생략하면
기본값 `0.1 0.4 0.5 1.0` 전체를 실행한다. 정확히 이 24개만 돌릴 때는 위의
cell/queue wrapper를 사용한다.

schema-v7 source audit가 image byte SHA-256에 의존하므로
`--no-verify_image_hashes`로 hash inventory를 생략하면 안 된다. 기본 launcher는
항상 `official_aod4_v6` 정책을 명시하며 overlap을 기록하되 purge하지 않는다.

## 6. Rank scaling

Rank 민감도에서 scaling을 바꾸면 rank 효과와 update scale 효과가 섞인다. 이
프로젝트의 LoRA forward scaling은 `lora_alpha/r`이며 다음처럼 고정한다.

| rank | lora_alpha | scaling |
|---:|---:|---:|
| 4 | 8 | 2 |
| 8 | 16 | 2 |
| 16 | 32 | 2 |

`run_rank_sensitivity.sh`가 이 값을 명시적으로 전달한다.

## 7. LoRA target ablation

Feature Extractor(backbone)까지 LoRA를 확장했다는 contribution을 분리해
검증하려면 decoder-only를 기준으로 backbone-only와 both를 꼭 비교해야
한다. 세 조건은 paired replicate 42/43/44 각각에서 동일한 s42/s43/s44
split을 공유하고, alpha=0.4, rank=8, `lora_alpha=16`, 20x5 budget을
사용한다.

```bash
bash scripts/run_target_ablation.sh
```

`both`는 primary `fl_fedsa_lora_r8_a0.4` 결과와 정확히 같은 조건이므로
그 결과를 재사용한다. `run_all_experiments.sh`는 기본
`RUN_TARGET_ABLATION=1`이며 primary 이후 decoder-only와 backbone-only만
추가 실행한다. 단순히 both가 decoder-only보다 높다는 결과만으로
일반적 backbone 효용을 단정하지 말고, seed 별 paired 차이와 통신·
학습 parameter 증가를 함께 보고한다.
각 target 조건에서도 AOD-4 task head는 학습·집계되므로, 이 ablation은
task head 유무가 아니라 LoRA adapter 주입 위치의 효과를 비교한다.
decoder target은 현재 구현의 decoder self-attention fused Q/K/V와
cross-attention value projection을 뜻한다. cross-attention의 Q/K까지 별도로
주입했다고 과장해 기술하지 않는다.

## 8. 결과 파일과 지표 정의

seed별 결과 예시:

```text
results/official_v6/seed_42/fl_full_ft_a0.4/fl_results.json
results/official_v6/seed_42/fl_lora_r8_a0.4/fl_results.json
results/official_v6/seed_42/fl_fedsa_lora_r8_a0.4/fl_results.json
results/official_v6/seed_42/fl_fixed_share_b_lora_r8_a0.4/fl_results.json
results/official_v6/seed_42/fl_fedsa_lora_r8_iid/fl_results.json
results/official_v6/seed_42/fl_fedsa_lora_r4_a0.4/fl_results.json
results/official_v6/seed_42/fl_fedsa_lora_r8_a0.4_decoder_only/fl_results.json
results/official_v6/seed_42/fl_fedsa_lora_r8_a0.4_backbone_only/fl_results.json
results/official_v6/seed_42/solo_lora_client0/solo_results.json
results/official_v6/seed_42/centralized_lora/centralized_results.json
```

검출 성능:

- `AP`: COCO-style AP, IoU 0.50:0.05:0.95 평균. Primary metric이다.
- `AP50`, `AP75`: IoU 0.50과 0.75의 AP.
- `avg_test_AP*`: client-local test AP의 macro 평균.
- `std_test_AP*`: client AP의 sample SD(`ddof=1`). seed 간 run-SD와 다른
  값이며 confidence interval이 아니다.
- `final_client_test`: client별 상세 AP.
- `cross_client_matrix`가 있으면 personalized model i를 client j test에
  평가한 행렬이다.
- `per_class`에는 class별 AP/AP50/AP75와 support가 함께 저장된다. 해당 client
  test에 존재하지 않는 class는 `absent_classes`로 명시되며 macro class AP의
  분모에 억지로 0으로 넣지 않는다.
- `common_test`는 모든 client가 같은 pooled test set에서 평가된 결과다.

COCO AP는 비선형 지표이므로 client AP의 평균과 pooled global-test AP는 같은
값이 아니다. client test에 특정 class가 없을 수도 있으므로 class별 support와
공통 pooled test 결과를 함께 확인해야 한다. client 수가 3개뿐인 표준편차는
불안정하므로 seed 3회의 mean/SD 또는 paired confidence interval을 별도로
보고한다.

파라미터와 통신:

- `total_params`: LoRA adapter를 포함한 현재 model parameter 수.
- `trainable_params`: 실제 `requires_grad=True` parameter 수. FedSA에서도 A와 B
  모두 학습하므로 일반 LoRA와 동일 계열이어야 한다.
- `comm_params`: client 한 곳의 한 방향 payload parameter 수.
- 이 필드는 엄밀히는 전송 state tensor element 수이며 full FT의
  non-parameter buffer도 포함될 수 있다.
- FedAvg-LoRA payload: LoRA A+B와 global AOD-4 task head.
- FedSA-LoRA payload: global LoRA A와 global AOD-4 task head. B는 local 유지.
- Fixed Share-B LoRA payload: global LoRA B와 global AOD-4 task head. A는 local 유지.
  A와 B의 tensor shape가 달라 두 대조군의 payload 크기는 같다고 가정하지 않고
  실제 element/byte를 각각 기록한다.
- `param_efficiency`: `100 x (1 - trainable_params/base_model_params)`.
- `parameter_counts.communication_saving_pct`: tensor element 수 기준
  `100 x (1 - communication_params/full_ft_payload_params)`.
- `communication.byte_saving_vs_full_ft_pct`: 실제 payload byte 기준 절감률이며
  논문 표의 primary communication efficiency다. top-level `comm_efficiency`와
  `comm_byte_saving_pct`도 이 byte 기준 alias이고, `comm_element_saving_pct`는
  element 기준 alias다. dtype이 다르면 두 절감률은 달라질 수 있다.
- 사용 비율은 `trainable_ratio_pct`, `communication.element_ratio_vs_full_ft_pct`,
  `communication.byte_ratio_vs_full_ft_pct`로 별도 저장된다.
- payload byte는 실제 tensor `numel x element_size`로 계산한다.
- `communication.one_client_one_way.mb`는 client 1개의 1회 단방향 payload,
  `communication.round_total.mb`는 모든 client의 upload+download 1 round 합계,
  `communication.cumulative_total.mb`는 실행된 전체 round 합계다.
- `per_round_comm_mb`는 `round_total.mb`, `total_comm_mb`는
  `cumulative_total.mb`의 flat alias이며 MB는 decimal `10^6 bytes`다.

한 round는 `N`개 client upload와 집계 뒤 `N`개 client download로 정의한다.
마지막 round의 download는 포함하고 최초 pretrained/base provisioning은
제외한다. 초기 checkpoint 배포, TCP/header, evaluation metric 전송은 현재
payload 계산에 포함하지 않는다. validation-selected best checkpoint가 마지막
round가 아닐 때 이를 다시 client에 배포하는 선택적 deployment 비용도 포함하지
않는다. 논문 표에는 이 convention을 적어야 한다.
personalized checkpoint에 재현/재개를 위해 모은 FedSA local B 또는 Fixed
Share-B local A state와 checkpoint file byte도 model-update 통신량에는 포함하지
않는다.
Centralized/solo의 FL model-update 통신은 full FT 대비 100% 절감으로 해석하지
말고 `N/A`로 구분한다. 중앙집중형의 원시 영상 이동 비용은 별도 개념이다.

학습 시각화 산출물은 experiment 디렉터리에 저장된다.

- `client_data_distribution.png`: client별 class/image 분포
- `training_loss_components.png`: 단독/중앙집중형 total/component loss
- `learning_rate_schedule.png`: 단독/중앙집중형 optimizer-group LR 범위
- `fl_training_curves.png`: round별 macro validation AP/AP50와 client sample-SD band
- `fl_loss_curves.png`: round별 client local loss
- `fl_loss_components.png`: effective local epoch별 RT-DETR GIoU/class/L1 loss
- `fl_learning_rate_schedule.png`: effective local epoch별 optimizer-group LR
- `detection_vis/`: 기본 5 epoch 또는 5 round마다 고정 표본의 validation 탐지 예시

시각화는 validation 데이터에 대한 진단용이며 test를 보고 hyperparameter나
checkpoint를 선택하지 않는다. `--visualize_interval 0`으로 끄더라도
성능 계산은 변하지 않아야 한다.

MIA:

- membership 단위는 annotation이 아니라 image다.
- member는 train image, non-member는 학습에 사용하지 않은 test image다.
- 기본 raw score는 ground-truth-matched RT-DETR image loss의 음수다. attack
  calibration subset의 AUC만 보고 lower-loss(`negative_detection_loss`) 또는
  higher-loss(`detection_loss`) 방향을 선택하며 evaluation label은 방향
  선택에 사용하지 않는 white-box loss-MIA baseline이다.
- `auc_roc`: threshold-independent attack ROC-AUC.
- `tpr_at_1fpr`: FPR 1% 이하에서 얻는 TPR.
- `asr`: attack calibration subset에서 Youden J로 선택한 threshold를
  서로 겹치지 않는 balanced evaluation subset에 적용한 attack accuracy.
- AUC/ASR 0.5, TPR@1%FPR 약 0.01이 random 수준이다. 방향은
  calibration에서 이미 고정되었으므로 evaluation AUC<0.5가 나와도 evaluation
  label을 보고 post-hoc로 반전하지 않는다. point estimate만으로 안전을
  단정하지 말고 `bootstrap_95_ci`를 함께 보고한다.
- `bootstrap_95_ci`는 기본 1,000회 stratified non-parametric resampling으로
  direction 선택, threshold calibration, evaluation을 모두 다시 수행한
  AUC/TPR@1%FPR/ASR 95% percentile interval이다.

`run_mia.sh`는 primary 결과 디렉터리의 checkpoint를 `--resume`하여
재학습 없이 같은 모델을 공격하고 결과 JSON의 `mia`에 덧쓴다. primary
결과가 없으면 fail-fast한다. primary만 먼저 학습하려면
`RUN_IID=0 RUN_ALPHA=0 RUN_RANK=0 RUN_TARGET_ABLATION=0 RUN_MIA=0`을
`run_all_experiments.sh`에 전달한다. 기본 `mia_max_samples=1000`은
client별 member와 non-member에 각각
적용된 상한이고, 작은 편의 표본 수로 정확히 balance한 뒤 기본 50%를
calibration으로 분리한다.

검출 AP는 공식 test 전체를 사용한다. MIA에서는 별도로, 어느 client의 train에든
존재하는 source component와 겹치는 test 이미지를 non-member 후보에서 제외한다.
제외 수와 최종 교집합 0 검증은 client별 `source_disjoint_sampling`에 기록된다.
이는 공식 검출 test를 삭제하는 처리가 아니다. 그 밖에도 MIA 표본은 class와
object size가 가능한 한 맞아야 한다. 그렇지 않으면 공격이 membership이 아니라
train/test source 차이를 검출할 수 있다. 작은
non-member 수에서 TPR@1%FPR는 불안정하므로 표본 수와 confidence interval을
함께 보고한다. 현재 loss-MIA는 최종 model에 대한 empirical audit이며 FL
server가 round별 individual update를 관찰하는 공격이나 gradient inversion을
대체하지 않는다.

### 8.1 Primary 결과를 변경하지 않는 MIA robustness audit

기존 `run_mia.sh`는 평가 결과를 primary `fl_results.json`에 다시 저장하므로,
강건성 분석 반복에는 사용하지 않는다. `mia_robustness_audit.py`는 seed
42, 43, 44에서 FL Full FT, FL LoRA, FedSA-LoRA, Fixed Share-B 네
validation-selected
`best_federated.pt`만 읽으며 결과와 loss cache를
seed 42는 `results/official_v6/security_audit_v1/`, seed 43·44는 각각
`security_audit_v1_seed43/`, `security_audit_v1_seed44/` 아래에만
기록한다. 기존 seed-42 audit를 덮어쓰지 않는다. 다음을 동시에
검증한다.

- primary `fl_results.json`, `best_federated.pt`, `last_federated.pt`, split
  manifest, pretrained weight의 실행 전/후 SHA-256 동일성
- checkpoint/primary의 선택 round, split digest, model-weight digest 및 전체
  architecture manifest(adapter target/shape, factor role, parameter count 포함)의
  exact digest 일치, method별 personalized factor role 복원
- RT-DETR-L, 4 classes, 640 px, batch 8, 20 rounds × 5 local epochs,
  optimizer/LR/scheduler, rank 8, alpha 16, both-targets 등 primary training
  compatibility manifest의 정확한 일치
- 동일한 hashed image/source-group ID를 사용한 네 방법의 paired attack pool과
  반복 attack split
- global train source group과 교집합이 없는 own-client test(non-member)를
  primary scope로, 동일 조건의 pooled test를 표본 해상도 확인용 secondary
  scope로 사용
- method-specific fresh four-class initialization loss와
  `delta_loss = trained_loss - initial_loss`를 이용한 official train/test
  split-origin 난이도 confounding 진단
- `selection_seed=420042`, `attack_seed=842042`, attack repeat 20회,
  calibration fraction 0.5, member/local/pooled cap 1000/1000/2000을 세
  replicate에서 같은 값으로 동결

먼저 출력 디렉터리를 만들지 않는 dry-run으로 seed 43·44의
모든 입력과 해시 계획을 확인한다.

```bash
cd /home/gpuadmin/kim/fedsalora
source .venv/bin/activate

AUDIT_SEED=43 bash scripts/run_mia_robustness_audit.sh dry-run
AUDIT_SEED=44 bash scripts/run_mia_robustness_audit.sh dry-run
```

실제 평가는 GPU 0·1에서 tmux로 병렬 실행한다. `AUDIT_SEED`만
바뀌며 나머지 protocol은 runner가 강제한다. `CUDA_VISIBLE_DEVICES=0`로
노출한 물리 GPU 0은 process 내부에서 `cuda`로 보인다.

```bash
tmux new-session -d -s mia_audit_s43 \
  "cd /home/gpuadmin/kim/fedsalora && \
   source .venv/bin/activate && \
   export CUDA_VISIBLE_DEVICES=0 && \
   export AUDIT_SEED=43 && \
   bash scripts/run_mia_robustness_audit.sh run"

tmux set-window-option -t mia_audit_s43:0 remain-on-exit on

tmux new-session -d -s mia_audit_s44 \
  "cd /home/gpuadmin/kim/fedsalora && \
   source .venv/bin/activate && \
   export CUDA_VISIBLE_DEVICES=1 && \
   export AUDIT_SEED=44 && \
   bash scripts/run_mia_robustness_audit.sh run"

tmux set-window-option -t mia_audit_s44:0 remain-on-exit on

tmux list-panes -a \
  -F 'session=#{session_name} dead=#{pane_dead} exit=#{pane_dead_status}' \
  | grep 'mia_audit_s4'
```

중단된 **동일한 v1 protocol**만 exact-key cache 검증 후 이어서 실행한다.
완료된 audit에는 `AUDIT_RESUME=1`을 사용해 덮어쓰지 말고 디렉터리 전체를
보존한다.

```bash
AUDIT_SEED=43 AUDIT_RESUME=1 bash scripts/run_mia_robustness_audit.sh run
```

주요 생성물은 `audit_report.json`, `audit_summary.csv`, `summary.md`, hashed
per-image loss record cache와 `integrity_manifest.json`이다. 파일명은 primary
집계기의 `*_results.json` 탐색 규칙과 충돌하지 않는다. loss record는 membership
정보를 포함하므로 directory mode 0700, file mode 0600으로 저장한다.

`trained_loss`는 GT annotation과 내부 RT-DETR training criterion을 사용하는
label-aware white-box final-checkpoint loss-threshold MIA score이다. `ASR`은
calibration subset에서 score 방향과 Youden-J threshold를 선택한 뒤, 균형을 맞춘
held-out evaluation member/non-member에서 계산한 balanced attack accuracy로
정의한다. AUC와 이 balanced accuracy의 chance baseline은 0.5이며,
0.5 근처의 차이나 0.5 미만의 sampling noise를 보안 우위로 순위화하지
않는다. `delta_loss`는 표준화된 FL-MIA가 아니라
**initialization-referenced loss-change diagnostic**이다. 즉 raw loss 순위가
이미지 난이도와 split-origin 차이에 민감한지 보는 통제·민감도 분석이며,
LiRA, differential privacy, formal confidentiality 보장으로 표현하지 않는다.

Robustness audit의 low-FPR 수치는 표준적인 evaluation ROC의
`TPR@1%FPR`와 구별해야 한다. calibration subset에서 FPR 목표 1%, 5%, 10% 이하인
threshold를 고정한 뒤 evaluation subset에 그대로 적용하며, 각각의 evaluation
TPR와 **실제 달성 FPR**, TP/FP 및 분모를 함께 기록한다. 따라서 논문에서는
`TPR at the calibration-targeted 1% FPR threshold (achieved evaluation FPR)`로
표기한다. 반복 split SD는 attack split 선택에 대한 강건성 분산이며, 독립적인
training-seed uncertainty나 confidence interval로 해석하지 않는다. v1은
source-group-atomic calibration/evaluation 분리를 적용하지만 cluster-bootstrap
CI, covariate-matched attack, confidence/entropy black-box attack 및 update-level
attack은 포함하지 않는다. 개인화 local factor까지 복원하므로 현재
위협 모델은 compromised/stolen client endpoint audit에 가깝다. honest-but-curious
server가 라운드별 client update에서 회원성을 추론하는 FL update-channel
attack을 대체하지 않는다.

seed 43·44 완료 후에는 client나 20개 attack split을 독립 표본으로
부풀리지 않고, 세 개의 paired `(training seed, partition seed)` audit만
통계 단위로 사용한다.

```bash
python3 scripts/summarize_mia_robustness.py \
  --results_root /home/gpuadmin/kim/fedsalora/results/official_v6

sed -n '1,240p' \
  /home/gpuadmin/kim/fedsalora/results/official_v6/\
security_audit_v1_multiseed_summary/summary.md
```

생성물은 `summary_long.csv`, `paired_method_differences.csv`,
`paired_trained_delta_differences.csv`, `summary.json`, `summary.md`,
`summary_manifest.json`이다. 표의 `±`는 세 paired replicate의 sample SD이며
attack-repeat SD가 아니다. training seed와 partition seed를 동시에 42/43/44로
변경했으므로, 이 SD는 순수 training-seed 분산으로 불러서는 안 된다.
집계기는 각 audit의 `before==after`, `changes=[]`, YOLO tree digest,
audit-report digest를 확인하고 보호된 primary JSON/checkpoint/split/weight를
현재 시점에 다시 SHA-256 검증한다. 또한 method별 primary training manifest가
seed 42/43/44 사이에 동일한지 확인한 뒤만 집계한다.

### 8.2 Covariate-matched read-only MIA audit

v1의 fresh-initialization negative control이 raw train/test 난이도 차이를 보였으므로,
추가 보안 분석은 새 공격을 무제한 추가하기보다 이 confounding을 직접 통제한다.
이 protocol은 해당 v1 결과를 확인한 뒤 설계한 **post-hoc exploratory sensitivity
analysis**이며, 사전등록된 confirmatory 보안 실험으로 기술하면 안 된다.
`mia_covariate_audit.py`는 seed 42/43/44의 기존 v1 per-image loss cache만 읽고,
member와 own-client nonmember를 다음 관측 공변량으로 outcome-blind coarsened-exact
matching한다.

- background 여부
- 클래스별 instance 구성(각 class count를 3에서 cap)
- 총 object count(6에서 cap)
- mean normalized bounding-box area의 log2 bin

매칭에는 trained/initial/delta loss를 전혀 사용하지 않는다. 매 repeat마다 먼저
member와 nonmember를 각각 source-group-atomic calibration/evaluation으로 나눈 뒤,
두 partition 안에서 별도로 매칭한다. 이 동일 plan을 네 방법과 세 score에 공통
적용한다. 각 calibration/evaluation partition에서 최소 64 matched pairs, 작은 pool
대비 retention 25% 이상, post-match max absolute SMD 0.10 이하를 사전에 고정한
fail-closed gate로 사용한다.
이 기준을 통과하지 못하면 결과를 본 뒤 임계값을 완화하지 말고 새 버전 protocol을
별도로 정의해야 한다.

이 작업은 checkpoint를 열거나 model forward를 실행하지 않는 CPU-only 재분석이다.
primary JSON/checkpoint/split/weight뿐 아니라 v1 report, integrity manifest와 사용한
모든 loss cache를 실행 전후 SHA-256으로 비교한다. 먼저 출력 디렉터리를 만들지
않는 dry-run을 수행한다.

```bash
cd /home/gpuadmin/kim/fedsalora
source .venv/bin/activate

python3 -m unittest tests.test_mia_covariate_audit
bash scripts/run_mia_covariate_audit.sh dry-run
```

dry-run이 `[PASS]`이면 실제 별도 결과를 생성한다.

```bash
bash scripts/run_mia_covariate_audit.sh run

sed -n '1,280p' \
  /home/gpuadmin/kim/fedsalora/results/official_v6/\
security_audit_v2_covariate_multiseed/summary.md
```

생성물은 `audit_report.json`, `summary_long.csv`, `paired_differences.csv`,
`matching_balance.csv`, `summary.md`, `integrity_manifest.json`이다. v1 입력과
완료된 v2 output은 덮어쓰지 않는다. matched `initial_loss`가 0.5 근처로
이동하고 trained-loss 차이가 유지되는지가 핵심 진단이다. 매칭 뒤 local
nonmember 수로는 1% FPR를 일관되게 분해할 수 없으므로 이 phase-2 표에서는
TPR/FPR@5%와 @10%만 보고한다. 기존 v1의 @1% 결과는 실제 달성 FPR와 함께 별도
진단으로 유지한다. 여기서 5%/10%는 calibration subset에서 목표 FPR로 선택한
임계값이며, 표에는 그 임계값을 evaluation subset에 적용한 TPR과 실제 달성 FPR을
함께 적는다. 다만 관측 annotation
공변량으로 포착되지 않는 scene/sensor/source 차이는 남을 수 있다. 이 결과도
endpoint label-aware loss attack에 한정되며 black-box confidence/entropy attack,
server update-channel attack, LiRA 또는 formal privacy guarantee가 아니다.
`paired_differences.csv`의 `nonnegative_chance_excess`는
`max(0, metric-0.5)`로 정의한 기술적 변환이며, 통상적인 MIA advantage
`TPR-FPR`를 뜻하지 않는다.

### 8.3 Seed-43/client-1 locally unseen helicopter 사례 추출

Primary split을 성능과 무관하게 먼저 검사하면, 42/43/44의 3-client train
partition 중 positive annotation support가 0인 셀은
`seed=43, client=1, class=helicopter` 한 건이다. 이는 일반적인 zero-shot이나
open-vocabulary 검출 실험이 아니라, Dirichlet 분할에서 자연스럽게 발생한
**client-local zero-positive-support case (`n=1`)**다. 새 학습이나 GPU 없이
기존 36개 primary JSON과 3개 immutable split manifest만 읽어 다음을 비교한다.

- Local Full FT/LoRA: client 1이 helicopter 양성 학습 box를 전혀 보지 않은
  local baseline
- Centralized Full FT/LoRA: pooled raw data를 직접 사용한 contextual ceiling
- FL Full FT/LoRA, FedSA-LoRA, Fixed Share-B: 다른 client의 helicopter 정보가
  서로 다른 parameter path를 통해 client-1 endpoint에 전달된 경우
- 각 방법에서 focal client에 대응하는 endpoint(비개인화 방법은 shared/global
  model)를 동일 common pooled test의 helicopter GT box 787개에서
  AP/AP50/AP75로 평가

서버에 새 파일을 복사한 뒤 다음을 실행한다. 이 작업은 GPU를 사용하지 않으므로
tmux가 필요하지 않다.

```bash
cd /home/gpuadmin/kim/fedsalora
source .venv/bin/activate

python3 -m unittest tests.test_summarize_unseen_class_case

python3 scripts/summarize_unseen_class_case.py \
  --results_root /home/gpuadmin/kim/fedsalora/results/official_v6 \
  --split_dir /home/gpuadmin/kim/fedsalora/data/splits \
  --output_dir /home/gpuadmin/kim/fedsalora/results/official_v6/\
summary_unseen_helicopter_s43_c1

sed -n '1,240p' \
  /home/gpuadmin/kim/fedsalora/results/official_v6/\
summary_unseen_helicopter_s43_c1/summary.md
```

집계기는 먼저 세 manifest만으로 유일한 zero-support 셀을 선택한 뒤 정확히
36개 canonical primary result를 검증한다. 100 epochs 또는 20 rounds x 5 local
epochs, RT-DETR-L/4 classes, rank 8/alpha 16/both-targets, optimizer와 LR schedule,
split SHA-256, 내장 split metadata, common-test support가 하나라도 다르면
output을 만들기 전에 fail-fast한다. 입력 39개의 실행 전/후 SHA-256은
`summary_manifest.json`에 남으며 primary JSON과 checkpoint는 수정하지 않는다.

생성물은 `unseen_class_case.csv`, `comparisons.csv`, `summary.json`, `summary.md`,
`summary_manifest.json`이다. 논문에서는 이를 extreme label-skew에서의
cross-client class-knowledge transfer를 보여 주는 기술 사례로만 사용한다.
단일 사례이므로 mean ± SD, confidence interval, significance 또는 일반적인
unseen-class robustness를 주장하지 않는다. 또한 모든 FL 방법이 task head를
공유하므로 FedSA와 Fixed Share-B의 차이를 A 또는 B factor 하나에만 귀속하지
않는다.

## 9. 학술적 해석 주의

1. FL은 원시 영상을 중앙 서버에 업로드하지 않는 data-locality 구조이지만 그
   자체가 보안 또는 프라이버시 보장은 아니다. model update와 최종 model에서
   membership, label, image 정보가 유출될 수 있다.
2. 보안 보장을 주장하려면 adversary model, secure aggregation, differential
   privacy와 privacy budget 등을 별도로 정의해야 한다. MIA가 낮다는 사실만으로
   기밀성이 증명되지는 않는다.
3. Dirichlet split은 synthetic label skew다. 실제 군사 surveillance site의
   camera, weather, background, sensor domain 차이를 직접 재현하지 않는다.
   `alpha=0.4`와 `0.5`는 가까운 조건이므로 명목 alpha만으로 차이를 단정하지
   말고 manifest의 realized class histogram과 JS divergence를 함께 보고한다.
4. schema-v7 Primary는 기존 논문과 비교하기 위해 공식 AOD-4 v6 split을
   보존하므로 source key/exact hash overlap도 그대로 남는다. video-derived
   frame의 near-duplicate나 시퀀스 상관까지 독립적이라고 주장하지 말고,
   perceptual/sequence-level audit를 후속 검증으로 다룬다.
5. FedSA의 “A=general, B=client-specific”은 NLP 원 논문의 가설을 CV로 옮긴
   것이다. 이 artifact는 동일 조건의 Fixed Share-B(global B/local A) 대조군을
   primary setting에 포함한다. 결과가 FedSA에 유리해도 보편적인 A/B 우열이
   아니라, 이 RT-DETR-LoRA parameterization과 고정 factor-sharing protocol에서의
   증거로 해석한다. cross-client swap·similarity 분석은 후속 검증이다.
6. 기대 결과를 확정된 결론처럼 쓰지 않는다. AP 개선, client 편차 감소, MIA
   감소는 seed 반복 후 검증할 연구 가설이다.
7. RT-DETR의 real-time 성격을 contribution에 포함하려면 동일 GPU, batch=1,
   같은 precision에서 preprocessing/postprocessing 포함 여부를 명시한 latency와
   FPS를 추가 측정해야 한다.
8. 현재 FL은 client/server를 한 process에서 실행하는 simulator이다. 실제
   network, authentication, transport encryption, client failure를 구현한 배포 시스템이
   아니다.
9. simulator의 personalized checkpoint는 재현과 `--resume`를 위해 모든
   client-local personalized factor(FedSA의 B, Fixed Share-B의 A)를 하나의 중앙
   checkpoint bundle에 저장한다. 이는 논문에서 주장하는 배포 protocol이 아니며,
   local factor state/checkpoint byte는 round model-update 통신량에서 제외된다.
   production에서는 해당 local factor를 각 client에 남겨야 한다.
10. `fixed_share_b_lora`는 공유 factor 역할을 반대로 둔 고정 대조군이며
    **FedAS-LoRA 구현이 아니다**. rank-aware adaptive factor sharing, 동적 선택,
    FedAS 논문의 adaptive rule은 이 코드에 넣지 않았고 related work 및 후속
    연구로만 다룬다.

## 10. 결과 집계

`run_all_experiments.sh`는 모든 seed 실험이 끝난 뒤 results root 전체에
집계 스크립트를 단 한 번 호출한다. 수동 실행은 다음과 같다.

```bash
python3 scripts/aggregate_results.py \
  /home/gpuadmin/kim/fedsalora/results/official_v6
```

생성물은 `results/summary/summary_by_method.csv`, `summary.md`,
`summary.json`과 `paired_fedsa_seed_deltas.csv`,
`paired_fedsa_comparisons.{csv,json,md}`와
`paired_factor_sharing_seed_deltas.csv`,
`paired_factor_sharing_comparisons.{csv,json,md}`다. 집계기는 학습 budget·optimizer·
split generator protocol·Ultralytics/pretrained-state fingerprint가 다른 결과를
같은 group으로 합치지 않는다. 각 replicate 내 method pairing은 실제 split
digest가 일치해야 하지만, replicate 간에는 같은 generator protocol의 다른
split realization을 하나의 run uncertainty로 집계한다. FedSA 대 FL full FT/LoRA는
동일 training seed와 partition seed, split digest에서 paired delta를 계산한다.
FedSA(global A/local B) 대 Fixed Share-B(global B/local A)도 rank, scaling,
LoRA target, optimizer/LR/budget이 같은 pair만 비교하며 성능과 실제 통신량 delta를
함께 출력한다.
schema-v7 grouping fingerprint에는 공식 split 보존 정책, source identity와 overlap
audit, 공식 split count 및 inventory/tree digest를 포함한다.
다만 manifest의 수만 개 per-file hash record 전체를 group key에 복사하지
않고 compact digest만 사용한다.
seed가 3개일 때 bootstrap 95%
interval은 탐색적·기술적 불확실성 표시이며 강한 유의성 검정으로 해석하지 않는다.
run-SD는 seed 42/43/44의 sample SD(`ddof=1`)다. client-SD와 run-SD를 표에서
명확히 구분한다. `status="complete"`가 아닌 중간/손상 결과는 재사용하지 않는다.
split manifest, 전체 CLI log, `environment.freeze.txt`, checkpoint와 JSON 결과를
함께 보존해야 완전한 재현이 가능하다.
> **Historical record / 과거 서버 실행 기록:** 이 문서는 원래 서버의 절대경로와 당시 실행 순서를 보존한 기록입니다. 새 환경에서 그대로 실행하지 마세요. 재현 절차는 [REPRODUCIBILITY.md](REPRODUCIBILITY.md), 데이터·split 제약은 [DATASET.md](DATASET.md)를 참고하세요.

#!/usr/bin/env python3
"""
0609auto_label.py  —  교통 표지판 데이터 수집 + 자동 라벨링 + YOLOv8n 학습 통합 스크립트
라즈베리파이 + IMX500 (Raspberry Pi AI Camera) 환경 전용.

실행하면 메뉴가 뜬다:
    [1] 촬영
    [2] 자동 라벨링  - dataset/images 안의 모든 jpg를 한 번에 라벨링
    [3] 모델 학습
    [q] 종료

파일명 규칙:
    {sign}_{light}_{dist}cm_{angle}_{NNN}.jpg

예:
    crosswalk_bright_30cm_front_001.jpg
    speed_30_dark_50cm_left_003.jpg

결과 구조:
    dataset/
      images/        촬영 이미지
      labels/        YOLO txt 라벨
      preview/       자동 라벨 검수용 이미지
      classes.txt    클래스 목록
      split/         학습용 train/val 분할 결과
      data.yaml      YOLO 학습 설정
"""

import os
import sys
import termios
import tty
import select
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────
# 조건 정의
# ──────────────────────────────────────────────────────────────────────────
SIGNS = {
    "1": ("stop", "Stop"),
    "2": ("do_not_enter", "Do Not Enter"),
    "3": ("school_zone", "School Zone"),
    "4": ("crosswalk", "Crosswalk"),
    "5": ("speed_30", "Speed Limit 30"),
    "6": ("speed_50", "Speed Limit 50"),
}

LIGHTS = {
    "1": "bright",
    "2": "dark",
}

ANGLES = {
    "1": "front",
    "2": "left",
    "3": "right",
}

DATASET_ROOT = Path("./dataset")
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
PREVIEW_DIR = DATASET_ROOT / "preview"

CLASSES = [v[0] for v in SIGNS.values()]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

CAPTURE_SIZE = (1280, 960)


# ──────────────────────────────────────────────────────────────────────────
# 자동 라벨링 모듈 import
# 같은 폴더에 auto_label.py가 있어야 한다.
# ──────────────────────────────────────────────────────────────────────────
try:
    import auto_label
    AUTO_LABEL_AVAILABLE = True
except ImportError:
    auto_label = None
    AUTO_LABEL_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════════════
def choose(prompt: str, options: dict):
    """
    번호 메뉴를 띄우고 선택을 받는다.
    'exit' 입력 시 None 반환.
    """
    print(f"\n{prompt}")

    for key, val in options.items():
        label = val[1] if isinstance(val, tuple) else val
        print(f"  [{key}] {label}")

    print("  (exit: 뒤로)")

    while True:
        sel = input("> ").strip().lower()

        if sel == "exit":
            return None

        if sel in options:
            val = options[sel]
            return val[0] if isinstance(val, tuple) else val

        print("잘못된 입력. 다시.")


def next_index(prefix: str) -> int:
    """
    같은 조건(prefix)으로 이미 찍은 이미지가 있으면 다음 번호부터 시작한다.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    mx = 0

    for f in IMAGES_DIR.glob(f"{prefix}*"):
        tail = f.stem[len(prefix):]

        if tail.isdigit():
            mx = max(mx, int(tail))

    return mx + 1


def write_classes_file():
    """
    dataset/classes.txt 생성.
    YOLO class_id 순서 확인용.
    """
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    (DATASET_ROOT / "classes.txt").write_text(
        "\n".join(CLASSES) + "\n",
        encoding="utf-8"
    )


def read_key_nonblocking(timeout=0.1):
    """
    터미널에서 키 하나를 즉시 읽는다.
    Enter를 누르지 않아도 P/Q 키 입력을 받을 수 있게 한다.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        r, _, _ = select.select([sys.stdin], [], [], timeout)

        if r:
            return sys.stdin.read(1)

        return None

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ══════════════════════════════════════════════════════════════════════════
# 촬영
# ══════════════════════════════════════════════════════════════════════════
def init_camera():
    """
    IMX500 카메라를 180도 회전 보정과 함께 초기화한다.
    """
    try:
        from picamera2 import Picamera2
        from libcamera import Transform

    except ImportError:
        sys.exit(
            "[오류] picamera2/libcamera 없음.\n"
            "라즈베리파이에서 아래 명령어 실행 후 다시 시도:\n"
            "sudo apt install -y python3-picamera2"
        )

    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={"size": CAPTURE_SIZE, "format": "RGB888"},
        transform=Transform(hflip=1, vflip=1),
    )

    picam2.configure(config)
    picam2.start()

    import time
    time.sleep(1.0)

    return picam2


def capture_session(picam2):
    """
    조건 선택 후 P키로 촬영.
    촬영 즉시 auto_label.py를 이용해서 라벨도 같이 생성한다.
    """
    while True:
        sign = choose("표지판 선택:", SIGNS)

        if sign is None:
            return

        light = choose("조명 조건:", LIGHTS)

        if light is None:
            continue

        angle = choose("촬영 각도:", ANGLES)

        if angle is None:
            continue

        dist = input("거리(cm) 입력 (예: 30): ").strip()

        if dist.lower() == "exit":
            continue

        if not dist.isdigit():
            print("거리는 숫자만 입력해야 함. 다시.")
            continue

        prefix = f"{sign}_{light}_{dist}cm_{angle}_"
        idx = next_index(prefix)

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)

        al_status = "자동 라벨링 ON" if AUTO_LABEL_AVAILABLE else "자동 라벨링 OFF (auto_label.py 없음)"

        print("\n" + "=" * 50)
        print(f"  촬영 모드: {sign} / {light} / {dist}cm / {angle}")
        print(f"  {al_status}")
        print("  [P] 즉시 촬영(+자동 박스)   |   [Q] 이 조건 종료")
        print("=" * 50)

        shot = 0

        while True:
            key = read_key_nonblocking(timeout=0.2)

            if key is None:
                continue

            key = key.lower()

            if key == "p":
                fname = f"{prefix}{idx:03d}.jpg"
                dest = IMAGES_DIR / fname

                picam2.capture_file(str(dest))

                label_msg = ""

                if AUTO_LABEL_AVAILABLE:
                    line, box = auto_label.auto_label_image(dest, CLASS_ID[sign])

                    if line:
                        lbl_path = LABELS_DIR / f"{prefix}{idx:03d}.txt"
                        lbl_path.write_text(line, encoding="utf-8")

                        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
                        auto_label.save_preview(dest, box, PREVIEW_DIR / fname)

                        label_msg = "  [라벨 OK]"
                    else:
                        label_msg = "  [라벨 실패 - 수동 필요]"

                print(f"  촬영 #{shot + 1}: {fname}{label_msg}")

                idx += 1
                shot += 1

            elif key == "q":
                print(f"  이 조건 종료. 총 {shot}장 촬영.\n")
                break

            elif key == "\x03":
                raise KeyboardInterrupt


def run_capture():
    """
    촬영 메뉴 실행.
    """
    write_classes_file()

    picam2 = init_camera()

    try:
        capture_session(picam2)

    except KeyboardInterrupt:
        print("\n촬영 중단.")

    finally:
        picam2.stop()
        print("카메라 종료.")

    print_summary()


# ══════════════════════════════════════════════════════════════════════════
# 일괄 자동 라벨링
# 주의: 이 함수들이 main()보다 위에 있어야 NameError가 안 난다.
# ══════════════════════════════════════════════════════════════════════════
def parse_class_from_filename(stem: str):
    """
    파일명 stem에서 클래스 이름을 추출한다.

    규칙:
        {sign}_{light}_{dist}cm_{angle}_{NNN}

    예:
        crosswalk_bright_30cm_front_001
        -> sign = crosswalk

        speed_30_bright_30cm_front_001
        -> sign = speed_30

        do_not_enter_dark_50cm_left_002
        -> sign = do_not_enter

    뒤에서 4개 토큰(light, dist, angle, number)을 제거한 나머지가 sign이다.
    """
    parts = stem.split("_")

    if len(parts) < 5:
        return None

    sign = "_".join(parts[:-4])

    if sign in CLASS_ID:
        return sign

    return None


def run_auto_label_all():
    """
    dataset/images/ 안에 있는 모든 jpg/jpeg/png 이미지를 자동 라벨링한다.
    이미 라벨 txt가 있으면 덮어쓰지 않고 건너뛴다.
    """
    if not AUTO_LABEL_AVAILABLE:
        print("[오류] auto_label.py가 현재 0609auto_label.py와 같은 폴더에 없음.")
        print("해결: auto_label.py 파일을 /home/girlgroup/EM2/finalproj/ 안에 넣어줘.")
        return

    if not IMAGES_DIR.is_dir():
        print("[중단] dataset/images/ 폴더가 없음. 먼저 [1] 촬영 진행.")
        return

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        imgs.extend(IMAGES_DIR.glob(ext))

    imgs = sorted(imgs)

    if not imgs:
        print("[중단] dataset/images/ 에 이미지가 없음. 먼저 [1] 촬영 진행.")
        return

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    total = len(imgs)
    ok = 0
    skip = 0
    fail = 0
    no_cls = 0

    print(f"\n이미지 {total}장 자동 라벨링 시작...\n")

    for i, img_path in enumerate(imgs, 1):
        lbl_path = LABELS_DIR / (img_path.stem + ".txt")

        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            skip += 1
            continue

        sign = parse_class_from_filename(img_path.stem)

        if sign is None:
            print(f"  [{i}/{total}] {img_path.name}  <- 파일명 규칙 불일치, 건너뜀")
            no_cls += 1
            continue

        line, box = auto_label.auto_label_image(img_path, CLASS_ID[sign])

        if line:
            lbl_path.write_text(line, encoding="utf-8")
            auto_label.save_preview(img_path, box, PREVIEW_DIR / img_path.name)

            print(f"  [{i}/{total}] {img_path.name}  [{sign}]  OK")
            ok += 1

        else:
            print(f"  [{i}/{total}] {img_path.name}  [{sign}]  박스 검출 실패")
            fail += 1

    print("\n" + "=" * 60)
    print("자동 라벨링 완료")
    print("=" * 60)
    print(f"성공: {ok}")
    print(f"건너뜀(기존 라벨 있음): {skip}")
    print(f"박스 검출 실패: {fail}")
    print(f"파일명 오류: {no_cls}")
    print(f"미리보기 폴더: {PREVIEW_DIR}")
    print("=" * 60 + "\n")

    if fail > 0:
        print("박스 검출 실패 사진은 배경이 복잡하거나 표지판이 너무 작을 수 있음.")
        print("dataset/preview/에서 성공한 박스가 잘 맞는지 꼭 확인해줘.")

    print_summary()


# ══════════════════════════════════════════════════════════════════════════
# 데이터 요약
# ══════════════════════════════════════════════════════════════════════════
def print_summary():
    """
    dataset/images 안의 이미지 수를 클래스와 조명 조건별로 요약한다.
    """
    from collections import defaultdict

    if not IMAGES_DIR.is_dir():
        print("아직 촬영 이미지 없음.")
        return

    files = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(IMAGES_DIR.glob(ext))

    by_sign = defaultdict(int)
    by_sl = defaultdict(int)

    for f in files:
        sign = parse_class_from_filename(f.stem)

        if sign is None:
            continue

        parts = f.stem.split("_")
        light = parts[-4]

        if sign in CLASSES and light in ("bright", "dark"):
            by_sign[sign] += 1
            by_sl[(sign, light)] += 1

    total = sum(by_sign.values())

    print("\n" + "=" * 50)
    print(f"  데이터 요약 (총 {total}장)")
    print("=" * 50)
    print(f"{'표지판':<16}{'bright':>8}{'dark':>8}{'합계':>8}")
    print("-" * 50)

    for s in CLASSES:
        b = by_sl.get((s, "bright"), 0)
        d = by_sl.get((s, "dark"), 0)
        t = by_sign.get(s, 0)
        flag = "  <-부족" if t < 100 else ""
        print(f"{s:<16}{b:>8}{d:>8}{t:>8}{flag}")

    print("-" * 50 + "\n")


# ══════════════════════════════════════════════════════════════════════════
# 모델 학습
# ══════════════════════════════════════════════════════════════════════════
def check_labels():
    """
    이미지마다 YOLO 라벨 txt가 있는지 확인한다.
    """
    if not IMAGES_DIR.is_dir():
        return 0, 0

    imgs = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        imgs.extend(IMAGES_DIR.glob(ext))

    labeled = 0

    for img in imgs:
        lbl = LABELS_DIR / (img.stem + ".txt")

        if lbl.exists() and lbl.stat().st_size > 0:
            labeled += 1

    return len(imgs), labeled


def make_data_yaml(split_dir: Path) -> Path:
    """
    YOLO 학습용 data.yaml 생성.
    """
    yaml_path = DATASET_ROOT / "data.yaml"

    content = (
        f"path: {split_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )

    yaml_path.write_text(content, encoding="utf-8")

    return yaml_path


def split_dataset(val_ratio=0.2):
    """
    images/와 labels/를 YOLO 표준 구조로 train/val 분할 복사한다.
    """
    import random
    import shutil

    split = DATASET_ROOT / "split"

    # 기존 split이 있어도 덮어쓰기 가능하게 폴더 생성
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (split / sub).mkdir(parents=True, exist_ok=True)

    imgs = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        imgs.extend(IMAGES_DIR.glob(ext))

    imgs = [
        f for f in imgs
        if (LABELS_DIR / (f.stem + ".txt")).exists()
        and (LABELS_DIR / (f.stem + ".txt")).stat().st_size > 0
    ]

    random.seed(42)
    random.shuffle(imgs)

    n_val = max(1, int(len(imgs) * val_ratio))
    val_set = set(imgs[:n_val])

    for img in imgs:
        part = "val" if img in val_set else "train"

        shutil.copy2(img, split / f"images/{part}" / img.name)

        lbl = LABELS_DIR / (img.stem + ".txt")
        shutil.copy2(lbl, split / f"labels/{part}" / lbl.name)

    return split, len(imgs) - n_val, n_val


def run_training():
    """
    YOLOv8n 학습 실행.
    """
    total, labeled = check_labels()

    print(f"\n이미지 {total}장, 라벨된 이미지 {labeled}장")

    if total == 0:
        print("[중단] 촬영된 이미지가 없음. 먼저 [1] 촬영을 진행해.")
        return

    if labeled == 0:
        print("\n" + "!" * 56)
        print("[중단] bounding box 라벨이 하나도 없음.")
        print("먼저 [2] 자동 라벨링을 실행해야 함.")
        print(f"클래스 순서는 dataset/classes.txt 와 동일하게: {CLASSES}")
        print("!" * 56 + "\n")
        return

    if labeled < 60:
        print(f"[경고] 라벨된 이미지가 {labeled}장으로 적음.")
        print("계속하려면 Enter, 중단은 exit 입력.")

        if input("> ").strip().lower() == "exit":
            return

    try:
        from ultralytics import YOLO

    except ImportError:
        print("[오류] ultralytics 없음.")
        print("설치 명령어:")
        print("pip install ultralytics")
        return

    print("\n데이터 분할 중 (train/val)...")
    split_dir, n_train, n_val = split_dataset()
    print(f"  train {n_train}장 / val {n_val}장")

    yaml_path = make_data_yaml(split_dir)

    print("\nYOLOv8n 학습 시작...")
    print("IMX500 변환을 고려해서 nano 모델인 yolov8n.pt 사용.\n")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(yaml_path),
        epochs=80,
        imgsz=640,
        batch=8,
        patience=20,
        project=str(DATASET_ROOT / "runs"),
        name="sign_yolov8n",
    )

    best = DATASET_ROOT / "runs" / "sign_yolov8n" / "weights" / "best.pt"

    print("\n" + "=" * 56)
    print("학습 완료!")
    print(f"가중치: {best}")
    print("\n다음 단계 — IMX500 변환 예시:")
    print("from ultralytics import YOLO")
    print(f"m = YOLO('{best}')")
    print("m.export(format='imx')")
    print("=" * 56 + "\n")


# ══════════════════════════════════════════════════════════════════════════
# 메인 메뉴
# main과 if __name__ == "__main__"는 반드시 파일 맨 아래쪽에 둔다.
# ══════════════════════════════════════════════════════════════════════════
def main():
    write_classes_file()

    while True:
        print("\n" + "#" * 50)
        print("  교통 표지판 파이프라인")
        print("#" * 50)
        print("  [1] 촬영")
        print("  [2] 자동 라벨링  (촬영한 이미지 전체 일괄 처리)")
        print("  [3] 모델 학습")
        print("  [q] 종료")

        sel = input("> ").strip().lower()

        if sel == "1":
            run_capture()

        elif sel == "2":
            run_auto_label_all()

        elif sel == "3":
            run_training()

        elif sel == "q":
            print("종료.")
            break

        else:
            print("1, 2, 3, q 중 선택.")


if __name__ == "__main__":
    main()

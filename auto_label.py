#!/usr/bin/env python3
"""
auto_label.py  —  단순 배경에서 표지판 bounding box 자동 검출 모듈

촬영 시 클래스(어떤 표지판인지)는 이미 알고 있으므로,
이 모듈은 "표지판이 사진 어디에 있는지" 위치만 찾으면 된다.

원리:
  단순 배경(벽/책상) 앞 표지판은 배경과 색·밝기가 뚜렷이 다르다.
  → 가장자리(엣지)를 찾고 → contour(윤곽선) 검출 → 가장 큰 윤곽의
    bounding box 를 표지판으로 본다.

거리별로 표지판이 작아져도, 배경이 단순하면 "배경과 다른 가장 큰 덩어리"가
표지판이므로 잘 잡힌다.

YOLO 라벨 형식:
  <class_id> <x_center> <y_center> <w> <h>
  모든 값은 0~1 사이로 정규화한다.
"""

import cv2
import numpy as np


def detect_sign_box(image, min_area_ratio=0.01, margin=0.02):
    """
    이미지에서 표지판으로 추정되는 가장 큰 객체의 bounding box를 찾는다.

    Args:
        image: BGR numpy 배열. cv2.imread()로 읽은 이미지.
        min_area_ratio: 이 비율보다 작은 덩어리는 무시한다.
                        화면의 1% 미만이면 표지판 아님으로 간주한다.
        margin: 박스를 살짝 키울 비율. 표지판 테두리 여유를 주기 위함.

    Returns:
        (x, y, w, h) 픽셀 좌표. 못 찾으면 None.
    """
    h_img, w_img = image.shape[:2]
    img_area = h_img * w_img

    # 1) 흑백 + 블러
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) Canny edge detection - 중앙값 기반 자동 임계값
    v = np.median(blur)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(blur, lo, hi)

    # 3) 끊긴 edge 연결
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # 4) contour 검출
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # 5) 너무 작은 contour 제거 후 가장 큰 contour 선택
    candidates = [
        c for c in contours
        if cv2.contourArea(c) >= img_area * min_area_ratio
    ]

    if not candidates:
        return None

    largest = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 6) margin 적용
    if margin > 0:
        dx = int(w * margin)
        dy = int(h * margin)

        x = max(0, x - dx)
        y = max(0, y - dy)

        x2 = min(w_img, x + w + 2 * dx)
        y2 = min(h_img, y + h + 2 * dy)

        w = x2 - x
        h = y2 - y

    return (x, y, w, h)


def box_to_yolo(box, img_w, img_h, class_id):
    """
    픽셀 bounding box를 YOLO 정규화 라벨 한 줄로 변환한다.

    Args:
        box: (x, y, w, h) 픽셀 좌표
        img_w: 이미지 너비
        img_h: 이미지 높이
        class_id: YOLO class id

    Returns:
        YOLO 라벨 문자열 1줄
    """
    x, y, w, h = box

    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h

    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"


def auto_label_image(image_path, class_id):
    """
    이미지 파일에서 박스를 찾아 YOLO 라벨 문자열을 반환한다.

    Args:
        image_path: 이미지 파일 경로
        class_id: YOLO class id

    Returns:
        (yolo_line, box) 또는 실패 시 (None, None)
    """
    img = cv2.imread(str(image_path))

    if img is None:
        return None, None

    box = detect_sign_box(img)

    if box is None:
        return None, None

    h_img, w_img = img.shape[:2]
    line = box_to_yolo(box, w_img, h_img, class_id)

    return line, box


def save_preview(image_path, box, out_path):
    """
    박스를 그린 미리보기 이미지를 저장한다.
    자동 라벨 검수용으로 dataset/preview/에 저장하는 것을 권장한다.

    Args:
        image_path: 원본 이미지 파일 경로
        box: (x, y, w, h)
        out_path: 저장할 미리보기 이미지 경로

    Returns:
        성공 True, 실패 False
    """
    img = cv2.imread(str(image_path))

    if img is None or box is None:
        return False

    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(str(out_path), img)

    return True


if __name__ == "__main__":
    """
    단독 테스트용 실행 방식:

    python auto_label.py <이미지경로> [class_id]

    예:
    python auto_label.py dataset/images/crosswalk_bright_30cm_front_001.jpg 3
    """
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("사용법: python auto_label.py <이미지경로> [class_id]")
        print("예시: python auto_label.py dataset/images/crosswalk_bright_30cm_front_001.jpg 3")
        sys.exit(0)

    path = Path(sys.argv[1]).expanduser()
    cid = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    line, box = auto_label_image(path, cid)

    if line:
        print(f"박스 검출됨: {box}")
        print(f"YOLO 라벨: {line.strip()}")

        preview_path = path.with_name(path.stem + "_preview.jpg")
        if save_preview(path, box, preview_path):
            print(f"미리보기 저장: {preview_path}")
    else:
        print("박스 못 찾음. 배경이 복잡하거나 표지판이 너무 작을 수 있음.")

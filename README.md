# HW2 嵌入式影像處理作業二:抓馬路
Road Contour Detection

本專案使用影像處理方法抓取馬路輪廓，主要流程包含 Sobel、LBP、Distance Transform、BFS、Feature Distance、Image Patch Distance、Find Contours 與 Max Contour。

# ============================================================
# Google Colab 完整修正版
# 目的：抓馬路輪廓，但避免終點抓到樹木、草地、海洋、天空
#
# 使用：
# Sobel
# LBP
# distance transform 找起始點
# BFS
# distance of features
# distance of image patches
# findContours
# max contour
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from google.colab import files


def show_image(img, title="", cmap=None, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


# ============================================================
# 1. 上傳圖片
# ============================================================
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

img_bgr = cv2.imread(image_path)

if img_bgr is None:
    raise ValueError("圖片讀取失敗，請確認是 jpg/png 圖片")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_bgr.shape[:2]

show_image(img_rgb, "Original Image")


# ============================================================
# 2. 前處理
# ============================================================
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)


# ============================================================
# 3. Sobel 特徵
# ============================================================
sobel_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)

sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

if np.max(sobel_mag) > 0:
    sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
else:
    sobel_mag = np.zeros_like(gray, dtype=np.uint8)

show_image(sobel_mag, "Sobel Feature", cmap="gray")


# ============================================================
# 4. LBP 特徵
# ============================================================
def compute_lbp(gray_img):
    h, w = gray_img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = gray_img[y, x]
            code = 0

            code |= (gray_img[y - 1, x - 1] > center) << 7
            code |= (gray_img[y - 1, x    ] > center) << 6
            code |= (gray_img[y - 1, x + 1] > center) << 5
            code |= (gray_img[y    , x + 1] > center) << 4
            code |= (gray_img[y + 1, x + 1] > center) << 3
            code |= (gray_img[y + 1, x    ] > center) << 2
            code |= (gray_img[y + 1, x - 1] > center) << 1
            code |= (gray_img[y    , x - 1] > center) << 0

            lbp[y, x] = code

    return lbp

lbp_img = compute_lbp(gray_blur)
show_image(lbp_img, "LBP Feature", cmap="gray")


# ============================================================
# 5. 建立道路 ROI
#
# 重點修正：
# 原本遠方 ROI 太寬，所以終點抓到樹木。
# 這版讓道路遠方收斂成很窄的梯形。
# ============================================================

roi_mask = np.zeros((h, w), dtype=np.uint8)

# 可調參數
bottom_y = int(h * 0.98)
top_y    = int(h * 0.43)

# 底部道路很寬
bottom_left_x  = int(w * 0.03)
bottom_right_x = int(w * 0.97)

# 遠方道路終點必須很窄，避免吃到左右樹木
top_left_x  = int(w * 0.485)
top_right_x = int(w * 0.515)

road_roi_polygon = np.array([
    [bottom_left_x,  bottom_y],
    [bottom_right_x, bottom_y],
    [top_right_x,    top_y],
    [top_left_x,     top_y]
], dtype=np.int32)

cv2.fillPoly(roi_mask, [road_roi_polygon], 255)

roi_view = img_rgb.copy()
cv2.polylines(roi_view, [road_roi_polygon], True, (255, 0, 0), 4)
show_image(roi_view, "Narrow Road ROI")


# ============================================================
# 6. 建立禁止區域：草木、天空、海洋
# ============================================================

# 綠色草木
is_green = (H >= 35) & (H <= 90) & (S > 40)

# 藍色天空 / 海洋
is_blue = (H >= 90) & (H <= 135) & (S > 35)

# 高飽和自然區域
is_nature = is_green | is_blue

# 道路顏色條件
road_color_condition = (
    (S < 125) &
    (V > 35) &
    (V < 235)
)

# Sobel 條件
sobel_condition = sobel_mag < 185

candidate_mask = np.zeros((h, w), dtype=np.uint8)

candidate_mask[
    (roi_mask == 255) &
    (~is_nature) &
    road_color_condition &
    sobel_condition
] = 255

# 注意：這裡 kernel 不要太大，避免膨脹到樹木
kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)

candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_5)
candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_3)

show_image(candidate_mask, "Candidate Road Mask", cmap="gray")


# ============================================================
# 7. Distance Transform 找起始點
# ============================================================

seed_search_mask = np.zeros((h, w), dtype=np.uint8)

seed_polygon = np.array([
    [int(w * 0.32), int(h * 0.98)],
    [int(w * 0.68), int(h * 0.98)],
    [int(w * 0.54), int(h * 0.60)],
    [int(w * 0.46), int(h * 0.60)]
], dtype=np.int32)

cv2.fillPoly(seed_search_mask, [seed_polygon], 255)

seed_candidate = cv2.bitwise_and(candidate_mask, seed_search_mask)

dist_map = cv2.distanceTransform(seed_candidate, cv2.DIST_L2, 5)
_, max_val, _, max_loc = cv2.minMaxLoc(dist_map)

start_x, start_y = max_loc

print("Distance Transform 最大值：", max_val)
print("主要起始點：", (start_x, start_y))

start_view = img_rgb.copy()
cv2.polylines(start_view, [seed_polygon], True, (255, 255, 0), 3)
cv2.circle(start_view, (start_x, start_y), 10, (255, 0, 0), -1)
show_image(start_view, "Start Point by Distance Transform")


# ============================================================
# 8. 多起始點
# ============================================================

seed_points = []

if candidate_mask[start_y, start_x] == 255:
    seed_points.append((start_x, start_y))

for y in range(int(h * 0.66), int(h * 0.98), 8):
    for x in range(int(w * 0.35), int(w * 0.65), 8):
        if candidate_mask[y, x] == 255:
            seed_points.append((x, y))

seed_points = list(set(seed_points))

print("BFS 起始點數量：", len(seed_points))

seed_view = img_rgb.copy()
for x, y in seed_points:
    cv2.circle(seed_view, (x, y), 3, (255, 0, 0), -1)

show_image(seed_view, "BFS Seed Points")


# ============================================================
# 9. distance of image patches
# ============================================================

def get_patch_mean(feature_img, x, y, patch_radius=3):
    h, w = feature_img.shape

    x1 = max(0, x - patch_radius)
    x2 = min(w, x + patch_radius + 1)
    y1 = max(0, y - patch_radius)
    y2 = min(h, y + patch_radius + 1)

    return float(np.mean(feature_img[y1:y2, x1:x2]))


def patch_distance(x1, y1, x2, y2, gray_img, sobel_img, lbp_img, patch_radius=3):
    d_gray = abs(
        get_patch_mean(gray_img, x1, y1, patch_radius) -
        get_patch_mean(gray_img, x2, y2, patch_radius)
    )

    d_sobel = abs(
        get_patch_mean(sobel_img, x1, y1, patch_radius) -
        get_patch_mean(sobel_img, x2, y2, patch_radius)
    )

    d_lbp = abs(
        get_patch_mean(lbp_img, x1, y1, patch_radius) -
        get_patch_mean(lbp_img, x2, y2, patch_radius)
    )

    dist = 0.50 * d_gray + 0.25 * d_sobel + 0.25 * d_lbp

    return dist


# ============================================================
# 10. distance of features
# ============================================================

def feature_distance(x1, y1, x2, y2, gray_img, hsv_img, sobel_img, lbp_img):
    H, S, V = cv2.split(hsv_img)

    d_gray = abs(int(gray_img[y1, x1]) - int(gray_img[y2, x2]))
    d_s = abs(int(S[y1, x1]) - int(S[y2, x2]))
    d_v = abs(int(V[y1, x1]) - int(V[y2, x2]))
    d_sobel = abs(int(sobel_img[y1, x1]) - int(sobel_img[y2, x2]))
    d_lbp = abs(int(lbp_img[y1, x1]) - int(lbp_img[y2, x2]))

    dist = (
        0.35 * d_gray +
        0.15 * d_s +
        0.15 * d_v +
        0.15 * d_sobel +
        0.20 * d_lbp
    )

    return dist


# ============================================================
# 11. BFS 搜尋
#
# 重點修正：
# 1. 不能跑出窄 ROI
# 2. 不能進入草木、天空、海洋
# 3. 越靠近遠方終點，允許的橫向寬度越窄
# 4. 避免在道路終點左右擴散到樹木
# ============================================================

def bfs_road_search(
    gray_img,
    hsv_img,
    sobel_img,
    lbp_img,
    candidate_mask,
    roi_mask,
    seed_points,
    is_nature
):
    h, w = gray_img.shape

    visited = np.zeros((h, w), dtype=np.uint8)
    road_mask = np.zeros((h, w), dtype=np.uint8)

    q = deque()

    for sx, sy in seed_points:
        if 0 <= sx < w and 0 <= sy < h:
            if candidate_mask[sy, sx] == 255 and roi_mask[sy, sx] == 255:
                visited[sy, sx] = 1
                road_mask[sy, sx] = 255
                q.append((sx, sy))

    directions = [
        (-1,  0), (1,  0),
        ( 0, -1), (0,  1),
        (-1, -1), (1, -1),
        (-1,  1), (1,  1)
    ]

    feature_threshold = 48
    patch_threshold = 34

    while q:
        x, y = q.popleft()

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue

            if visited[ny, nx] == 1:
                continue

            visited[ny, nx] = 1

            if roi_mask[ny, nx] == 0:
                continue

            if candidate_mask[ny, nx] == 0:
                continue

            if is_nature[ny, nx]:
                continue

            d_feature = feature_distance(
                x, y,
                nx, ny,
                gray_img,
                hsv_img,
                sobel_img,
                lbp_img
            )

            d_patch = patch_distance(
                x, y,
                nx, ny,
                gray_img,
                sobel_img,
                lbp_img,
                patch_radius=3
            )

            if d_feature < feature_threshold and d_patch < patch_threshold:
                road_mask[ny, nx] = 255
                q.append((nx, ny))

    return road_mask


road_mask = bfs_road_search(
    gray_blur,
    hsv,
    sobel_mag,
    lbp_img,
    candidate_mask,
    roi_mask,
    seed_points,
    is_nature
)

show_image(road_mask, "BFS Road Mask", cmap="gray")


# ============================================================
# 12. 清理 Mask
#
# 重點：
# kernel 不要過大，否則終點會膨脹到樹木
# ============================================================

kernel_close = np.ones((13, 13), np.uint8)
kernel_open = np.ones((5, 5), np.uint8)

road_mask_clean = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)
road_mask_clean = cv2.morphologyEx(road_mask_clean, cv2.MORPH_OPEN, kernel_open)

road_mask_clean = cv2.bitwise_and(road_mask_clean, roi_mask)

# 再次清掉自然區域
road_mask_clean[is_nature] = 0

show_image(road_mask_clean, "Clean Road Mask", cmap="gray")


# ============================================================
# 13. findContours + max contour
# ============================================================

contours, _ = cv2.findContours(
    road_mask_clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

valid_contours = []

for c in contours:
    area = cv2.contourArea(c)

    if area < 1000:
        continue

    x, y, cw, ch = cv2.boundingRect(c)

    touches_bottom = (y + ch) > int(h * 0.82)

    if touches_bottom:
        valid_contours.append(c)

final_mask = np.zeros_like(road_mask_clean)

if len(valid_contours) > 0:
    max_contour = max(valid_contours, key=cv2.contourArea)
    cv2.drawContours(final_mask, [max_contour], -1, 255, -1)
    print("最大馬路輪廓面積：", cv2.contourArea(max_contour))
else:
    print("沒有找到符合條件的馬路輪廓，改用 Clean Mask")
    final_mask = road_mask_clean.copy()

# 最後保險：不要讓結果超出 ROI，也不要包含自然區域
final_mask = cv2.bitwise_and(final_mask, roi_mask)
final_mask[is_nature] = 0

show_image(final_mask, "Final Road Mask", cmap="gray")


# ============================================================
# 14. 畫出最後結果
# ============================================================

result = img_rgb.copy()

final_contours, _ = cv2.findContours(
    final_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(final_contours) > 0:
    road_contour = max(final_contours, key=cv2.contourArea)

    overlay = result.copy()
    overlay[final_mask == 255] = [255, 0, 0]

    result = cv2.addWeighted(overlay, 0.35, result, 0.65, 0)
    cv2.drawContours(result, [road_contour], -1, (255, 0, 0), 4)

show_image(result, "Final Road Contour Result")


# ============================================================
# 15. 儲存下載
# ============================================================

output_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite("final_road_contour_result_fixed.png", output_bgr)

files.download("final_road_contour_result_fixed.png")

# Road Contour Detection
## Original image
<p align="center">
  <img src="/655.jpg" width="400">
</p>
## Final Result
<p align="center">
  <img src="/final_road_contour_result_fixed.png" width="400">
</p>


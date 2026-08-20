# run_viola-jones.py

import cv2
import numpy as np
#import matplotlib      # บรรทัด 5-6 ใช้อันนี้ถ้ารันแล้วยังเกิด Error อยู่ <เก็บไว้เป็นแผนสำรอง> 
#matplotlib.use('Agg')  # ถ้าใช้ตัวเดิมได้ อันนี้ไม่ต้องใช้ลบตัวนี้ทิ้งได้เลย
import matplotlib.pyplot as plt

def haar_edge_h(ii, x, y, w, h):
    half = h // 2
    white = rect_sum(ii, x, y, x + w - 1, y + half - 1)
    black = rect_sum(ii, x, y + half, x + w - 1, y + h - 1)
    return white - black


def haar_edge_v(ii, x, y, w, h):
    half = w // 2
    white = rect_sum(ii, x, y, x + half - 1, y + h - 1)
    black = rect_sum(ii, x + half, y, x + w - 1, y + h - 1)
    return white - black


def haar_line_v(ii, x, y, w, h):
    third = w // 3
    left = rect_sum(ii, x, y, x + third - 1, y + h - 1)
    mid = rect_sum(ii, x + 2 * third, y, x + w - 1, y + h - 1)
    right = rect_sum(ii, x + 2 * third, y, x + w - 1, y + h - 1)
    return (left + right) - mid


def haar_four(ii, x, y, w, h):
    hw, hh = w // 2, h // 2
    tl = rect_sum(ii, x, y, x + hw - 1, y + hh - 1)
    tr = rect_sum(ii, x + hw, y, x + w - 1, y + hh - 1)
    bl = rect_sum(ii, x, y + hh, x + hw - 1, y + h - 1)
    br = rect_sum(ii, x + hw, y + hh, x + w - 1, y + h - 1)
    return (tl + br) - (tr + bl)

def haar_feature_value(ii, x, y, w, h, kind):
    if kind == "edge_h":
        return haar_edge_h(ii, x, y, w, h)
    elif kind == "edge_v":
        return haar_edge_v(ii, x, y, w, h)
    elif kind == "line_v":
        return haar_line_v(ii, x, y, w, h)
    elif kind == "four":
        return haar_four(ii, x, y, w, h)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def build_integral_image(gray):
    ii = np.cumsum(np.cumsum(gray.astype(np.float64), axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
    return ii


def rect_sum(ii, x1, y1, x2, y2):
    return ii[y2 + 1, x2 + 1] - ii[y1, x2 + 1] - ii[y2 + 1, x1] + ii[y1, x1]


def weak_classify(value, threshold, polarity):
    return 1 if polarity * value < polarity * threshold else 0


def best_threshold_for_feature(values, labels, weights):
    order = np.argsort(values)
    v, y, w = values[order], labels[order], weights[order]

    total_pos = np.sum(w[y == 1])
    total_neg = np.sum(w[y == 0])

    cum_pos = np.cumsum(np.where(y == 1, w, 0))
    cum_neg = np.cumsum(np.where(y == 0, w, 0))

    # Error when polarity = +1
    err_p1 = cum_neg + (total_pos - cum_pos)
    # Error when polarity = -1
    err_m1 = cum_pos + (total_neg - cum_neg)

    i1 = np.argmin(err_p1)
    i2 = np.argmin(err_m1)

    if err_p1[i1] <= err_m1[i2]:
        return v[i1], 1, err_p1[i1]
    else:
        return v[i2], -1, err_m1[i2]


def build_feature_bank(n_features=60, win=24, seed=42):
    rng = np.random.default_rng(seed)
    kinds = ["edge_h", "edge_v", "line_v", "four"]
    bank = []

    for _ in range(n_features):
        kind = rng.choice(kinds)
        w = int(rng.integers(win // 3, win + 1))
        h = int(rng.integers(win // 3, win + 1))

    # adjustsize todivisible
    if kind == "line_v":
            w -= w % 3
            w = max(w, 3)
    else:
            w -= w % 2
            h -= h % 2
            w, h = max(w, 2), max(h, 2)

    x = int(rng.integers(0, win - w + 1))
    y = int(rng.integers(0, win - h + 1))
    bank.append((kind, x, y, w, h))

    return bank


def extract_all_features(ii_win, feature_bank):
    return np.array([haar_feature_value(ii_win, x, y, w, h, kind)
                    for (kind, x, y, w, h) in feature_bank])


def train_adaboost(X, labels, n_rounds):
    n = len(labels)
    weights = np.full(n, 1.0 / n)  # initial weights are equal
    ensemble = []

    for rnd in range(n_rounds):
        weights /= weights.sum()  # Normalize
        best = (None, None, None, np.inf, None)

    # test every feature
    for f in range(X.shape[1]):
            thr, pol, err = best_threshold_for_feature(X[:, f], labels, weights)
            if err < best[3]:
                best = (f, thr, pol, err, None)

    f_idx, thr, pol, err, _ = best
    err = np.clip(err, 1e-10, 1 - 1e-10)

    # calculate Alpha: α = ½·ln((1-ε)/ε)
    alpha = 0.5 * np.log((1 - err) / err)
    ensemble.append((f_idx, thr, pol, alpha))

    # updateweight
    preds = np.array([weak_classify(v, thr, pol) for v in X[:, f_idx]])
    correct = preds == labels
    weights[correct] *= np.exp(-alpha)   # correct -> decrease weight
    weights[~correct] *= np.exp(alpha)   # incorrect -> increase weight

    print(f" Round {rnd+1}: feature#{f_idx}, threshold={thr:.4f}, "
          f"polarity={pol}, error={err:.4f}, alpha={alpha:.4f}")

    return ensemble


def strong_classify(ensemble, feature_values):
    total = sum(alpha * weak_classify(feature_values[f], thr, pol)
                for f, thr, pol, alpha in ensemble)
    alpha_sum = sum(alpha for _, _, _, alpha in ensemble)
    return 1 if total >= 0.5 * alpha_sum else 0


def train_cascade(X, labels, stage_rounds=(3, 6)):
    cascade = []
    for i, n_rounds in enumerate(stage_rounds):
        print(f"\n--- Training Stage {i+1} ({n_rounds} rounds) ---")
        ensemble = train_adaboost(X, labels, n_rounds)
        cascade.append(ensemble)
    return cascade


def cascade_classify(cascade, feature_values):
    for stage in cascade:
        if strong_classify(stage, feature_values) == 0:
            return 0  # ❌ Reject
    return 1  # ✅ Yesface


def sliding_window_detection(img_gray, cascade, feature_bank, win=24, step=4):
    H, W = img_gray.shape
    detections = []
    total_windows = ((H - win) // step) * ((W - win) // step)

    for y in range(0, H - win, step):
        for x in range(0, W - win, step):
            # reject Sub-window
            crop = img_gray[y:y + win, x:x + win]

            # create Integral Image
            ii_crop = build_integral_image(crop)

            # extract Features
            fvals = extract_all_features(ii_crop, feature_bank)

            # checkusing Cascade
            if cascade_classify(cascade, fvals) == 1:
                detections.append((x, y, win, win))

    return detections


def merge_boxes(boxes, win):
    if not boxes:
        return []

    boxes = np.array(boxes)
    centers = boxes[:, :2] + win // 2
    used = np.zeros(len(boxes), dtype=bool)
    merged = []

    for i in range(len(boxes)):
        if used[i]:
            continue
        close = np.linalg.norm(centers - centers[i], axis=1) < win
        group = boxes[close & ~used]
        used[close] = True
        gx = int(np.mean(group[:, 0]))
        gy = int(np.mean(group[:, 1]))
        merged.append((gx, gy, win, win))

    return merged

def make_synthetic_face(size=24, noise=0):
    img = np.full((size, size), 190, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    eg = size // 4

    # round face
    cv2.ellipse(img, (cx, cy), (size // 2 - 2, size // 2 - 1), 0, 0, 360, 205, -1)

    # eyes (dark)
    cv2.circle(img, (cx - eg, cy - 2), max(2, size // 10), 40, -1)
    cv2.circle(img, (cx + eg, cy - 2), max(2, size // 10), 40, -1)

    # nose
    cv2.line(img, (cx, cy + 1), (cx, cy + size // 5), 150, 1)

    if noise:
        img = img.astype(np.int16) + np.random.randint(-noise, noise + 1, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def make_synthetic_background(size=24, seed=None):
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 3)
    img = np.full((size, size), int(rng.integers(60, 220)), dtype=np.uint8)

    if kind == 0:  # line pattern
        step = rng.integers(3, 8)
        for i in range(0, size, step):
            cv2.line(img, (0, i), (size, i), int(rng.integers(30, 230)), 1)
    elif kind == 1:  # rectangles
        for _ in range(4):
            pt1 = tuple(rng.integers(0, size, 2))
            pt2 = tuple(rng.integers(0, size, 2))
            cv2.rectangle(img, pt1, pt2, int(rng.integers(30, 230)), -1)
    else:  # Noise
        img = rng.integers(0, 255, (size, size), dtype=np.uint8)

    return img


def build_training_set(n_pos=40, n_neg=60, win=24):
    rng = np.random.default_rng(0)
    positives, negatives = [], []

    for _ in range(n_pos):
        cx = win // 2 + int(rng.integers(-1, 2))
        cy = win // 2 + int(rng.integers(-1, 2))
        eg = win // 4 + int(rng.integers(-1, 2))
        positives.append(make_synthetic_face(win, noise=8))

    for i in range(n_neg):
        negatives.append(make_synthetic_background(win, seed=i))

    X_imgs = positives + negatives
    y = np.array([1] * n_pos + [0] * n_neg)
    return X_imgs, y


def create_dataset(win):
    print("\n[1] Create synthetic dataset...")

    imgs, labels = build_training_set(
        n_pos=40,
        n_neg=60,
        win=win
    )

    print(f"  Face images: {np.sum(labels == 1)}")
    print(f"  Non-face images: {np.sum(labels == 0)}")

    return imgs, labels


def create_features(win):
    print("\n[2] Build Feature Bank...")

    feature_bank = build_feature_bank(
        n_features=60,
        win=win
    )

    print(f"  Number of features: {len(feature_bank)}")

    return feature_bank

def extract_features(imgs, feature_bank):
    print("\n[3] Calculate features...")

    X = np.zeros(
        (len(imgs), len(feature_bank))
    )

    for i, im in enumerate(imgs):
        ii = build_integral_image(im)
        X[i] = extract_all_features(
            ii,
            feature_bank
        )

    print(f"  Feature matrix: {X.shape}")

    return X


def train_model(X, labels):
    print("\n[4] Train Cascade...")

    cascade = train_cascade(
        X,
        labels,
        stage_rounds=(3, 8)
    )

    print(f"  Number of stages: {len(cascade)}")

    return cascade


def test_model(cascade, feature_bank, win):
    print("\n[5] Test on synthetic image...")

    test_img = make_synthetic_face(
        win,
        noise=5
    )

    ii = build_integral_image(test_img)

    fvals = extract_all_features(
        ii,
        feature_bank
    )

    result = cascade_classify(
        cascade,
        fvals
    )

    status = "Face" if result == 1 else "Non-face"

    print(f"  Result: {status}")


def detect_faces(cascade, feature_bank, win, step):
    print("\n[6] Detect faces in real image...")

    image_path = "lena_gray-1.png" # ที่ใส่ชื่อภาพไว้ที่จะตรวจจับ

    img_gray = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img_gray is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    img_gray = cv2.resize(
        img_gray,
        (256, 256)
    )

    print(f"  Loaded image: {image_path}")

    detections = sliding_window_detection(
        img_gray,
        cascade,
        feature_bank,
        win,
        step
    )

    merged = merge_boxes(
        detections,
        win
    )

    print(f"  Detected faces: {len(merged)}")

    save_result(
        img_gray,
        merged
    )

def save_result(img_gray, boxes):
    print("\n[7] Save and display results...")

    result = cv2.cvtColor(
        img_gray,
        cv2.COLOR_GRAY2BGR
    )

    for x, y, w, h in boxes:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    output_path = "viola_jones_result.png"

    cv2.imwrite(
        output_path,
        result
    )

    print(f"  Saved image: {output_path}")

    plt.figure(figsize=(8, 8))

    plt.imshow(
        cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"Viola-Jones Face Detection - "
        f"{len(boxes)} faces detected"
    )

    plt.axis("off")
    plt.show()


def main():
    WIN = 117 # Default 24 <ปรับค่าสี่เหลี่ยมตรวจจับ> ยิ่งปรับค่ามาก สี่เหลี่ยมที่ตรวจจับจะน้อยลงตามค่าที่ปรับ
    STEP = 4

    print("=" * 70)
    print("VIOLA-JONES FACE DETECTION")
    print("=" * 70)

    imgs, labels = create_dataset(WIN)
    feature_bank = create_features(WIN)
    X = extract_features(imgs, feature_bank)
    cascade = train_model(X, labels)
    test_model(cascade, feature_bank, WIN)
    detect_faces(cascade, feature_bank, WIN, STEP)

    print("\nFinished!")

if __name__ == "__main__":
    main()

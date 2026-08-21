# run_viola-jones.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def haarEdgeHorizontal(integralImage, x, y, width, height):
    
    half = height // 2 # หารเอาส่วน

    white = rectangleSum(
        integralImage,
        x,
        y,
        x + width - 1,
        y + half - 1
    )

    black = rectangleSum(
        integralImage,
        x + half,
        y,
        x + width - 1,
        y + height - 1
    )

    return white - black


def haarEdgeVertical(integralImage, x, y, width, height):

    half = width // 2

    white = rectangleSum(
        integralImage,
        x,
        y,
        x + half - 1,
        y + height - 1
    )

    black = rectangleSum(
        integralImage,
        x + half,
        y,
        x + width - 1,
        y + height - 1
    )

    return white - black


def haarLineVertical(integralImage, x, y, width, height):

    third = width // 3

    left = rectangleSum(
        integralImage,
        x,
        y,
        x + third - 1,
        y + height - 1
    )

    mid = rectangleSum(
        integralImage,
        x + third,
        y,
        x + 2 * third - 1,
        y + height - 1
    )

    right = rectangleSum(
        integralImage,
        x + 2 * third,
        y,
        x + width - 1,
        y + height - 1
    )

    return (left + right) - mid


def haarFour(integralImage, x, y, width, height):

    # t left & t right | b left & b right

    halfWidth, halfHeight = width // 2, height // 2

    tLeft = rectangleSum(
        integralImage,
        x,
        y,
        x + halfWidth - 1,
        y + halfHeight - 1
    )
    
    tRight = rectangleSum(
        integralImage,
        x + halfWidth,
        y,
        x + width - 1,
        y + halfHeight - 1
    )

    bLeft = rectangleSum(
        integralImage,
        x,
        y + halfHeight,
        x + halfWidth - 1,
        y + height - 1
    )

    bRight = rectangleSum(
        integralImage,
        x + halfWidth,
        y + halfHeight,
        x + width - 1,
        y + height - 1
    )

    return (tLeft + bRight) - (tRight + bLeft)


def haarFeatureValue(integralImage, x, y, width, height, kind):

    if kind == "edgeHorizontal":
        return haarEdgeHorizontal(integralImage, x, y, width, height)
    
    elif kind == "edgeVertical":
        return haarEdgeVertical(integralImage, x, y, width, height)

    elif kind == "lineVertical":
        return haarLineVertical(integralImage, x, y, width, height)

    elif kind == "four":
        return haarFour(integralImage, x, y, width, height)

    else:
        raise ValueError(f"Unknown kind: {kind}")


def buildIntegralImage(gray):

    integralImage = np.cumsum(np.cumsum(gray.astype(np.float64), axis = 0), axis = 1)
    integralImage = np.pad(integralImage, ((1, 0), (1, 0)), mode = "constant")

    return integralImage


def rectangleSum(integralImage, x1, y1, x2, y2):

    return integralImage[y2 + 1, x2 + 1] - integralImage[y1, x2 + 1] - integralImage[y2 + 1, x1] + integralImage[y1, x1]


def weakClassify(value, threshold, polarity):

    return 1 if polarity * value < polarity * threshold else 0

def bestThresholdForFeature(values, labels, weights):

    order = np.argsort(values)
    value, y, width = values[order], labels[order], weights[order]

    totalPos = np.sum(width[y == 1])
    totalNeg = np.sum(width[y == 0])

    cumPos = np.cumsum(np.where(y == 1, width, 0))
    cumNeg = np.cumsum(np.where(y == 0, width, 0))

    # error when polarity = +1
    errPos1 = cumNeg + (totalPos - cumPos)

    # error when polarity = -1
    errNeg1 = cumPos + (totalNeg - cumNeg)

    index1 = np.argmin(errPos1)
    index2 = np.argmin(errNeg1)

    if errPos1[index1] <= errNeg1[index2]:

        return value[index1], 1, errPos1[index1]

    else:

        return value[index2], -1, errNeg1[index2]


def buildFeatureBank(nFeature = 60, window = 24, seed = 42):

    rng = np.random.default_rng(seed)
    kinds = ["edgeHorizontal", "edgeVertical", "lineVertical", "four"]
    bank = []

    for _ in range(nFeature):

        kind = rng.choice(kinds)
        width = int(rng.integers(window // 3, window + 1))
        height = int(rng.integers(window // 3, window + 1))

        # adjust size to divisible

        if kind == "lineVertical":

            width -= width % 3
            width = max(width, 3)

        else:

            width -= width % 2
            height -= height % 2
            width, height = max(width, 2), max(height, 2)

        x = int(rng.integers(0, window - width + 1))
        y = int(rng.integers(0, window - height + 1))
        bank.append((kind, x, y, width, height))

    return bank


def extractAllFeature(integralImageWindow, featureBank):

    return np.array([haarFeatureValue(integralImageWindow, x, y, width, height, kind)
                     for (kind, x, y, width, height) in featureBank])


def trainAdaBoost(x, labels, nRound):

    n = len(labels)
    weight = np.full(n, 1.0 / n) # initail weight are equal
    ensemble = []

    for round in range(nRound):

        weight /= weight.sum() # normalizing
        best = (None, None, None, np.inf, None)

        # test every feature

        for f in range(x.shape[1]):

            threshold, polarity, err = bestThresholdForFeature(x[:, f], labels, weight)

            if err < best[3]:

                best = (f, threshold, polarity, err, None)

        fIndex, threshold, polarity, err, _ = best
        err = np.clip(err, 1e-10, 1 - 1e-10)

        # calculate alpha: α = 1 / 2 * ln((1 - ε) / ε)
        alpha = 0.5 * np.log((1 - err) / err)
        ensemble.append((fIndex, threshold, polarity, alpha))

        # update weight
        predict = np.array([weakClassify(value, threshold, polarity) for value in x[:, fIndex]])
        correct = predict == labels
        weight[correct] *= np.exp(-alpha) # correct -> decrease weight
        weight[~correct] *= np.exp(alpha) # incorrect -> increase weight

        print(f"\tround {round + 1}: feature #{fIndex}, threshold = {threshold:.4f}, "
              f"polarity = {polarity}, error = {err:.4f}, alpha = {alpha:.4f}")

    return ensemble


def strongClassify(ensemble, featureValue):

    total = sum(alpha * weakClassify(featureValue[f], threshold, polarity)
                for f, threshold, polarity, alpha in ensemble)

    alphaSum = sum(alpha for _, _, _, alpha in ensemble)

    return 1 if total >= 0.5 * alphaSum else 0

def trainCascade(x, labels, stageRound = (3, 6)): # default: x, labels, stageRound = (3, 6)

    cascade = []
    for i, nRound in enumerate(stageRound):

        print(f"\nTraining stage {i + 1} ({nRound} Rounds)")
        ensemble = trainAdaBoost(x, labels, nRound)
        cascade.append(ensemble)

    return cascade


def cascadeClassify(cascade, featureValue):

    for stage in cascade:

        if strongClassify(stage, featureValue) == 0:

            return 0 # it's not a face, reject

    return 1 # it's a face, approve


def slidingWindowDetection(imageGray, cascade, featureBank, window = 24, step = 4):

    height, width = imageGray.shape
    detection = []
    totalWindow = ((height - window) // step) * ((width - window) // step)

    for y in range(0, height - window, step):

        for x in range(0, width - window, step):

            # reject sub-window

            crop = imageGray[y:y + window, x:x + window]

            # create integral image

            integralImageCrop = buildIntegralImage(crop)

            # extract feature

            featureValue = extractAllFeature(integralImageCrop, featureBank)

            # check using cascade

            if cascadeClassify(cascade, featureValue) == 1:

                detection.append((x, y, window, window))

    return detection


def mergeBox(box, window):

    if not box:

        return []

    box = np.array(box)
    center = box[:, :2] + window // 2
    used = np.zeros(len(box), dtype = bool)
    merged = []

    for i in range(len(box)):

        if used[i]:

            continue

        close = np.linalg.norm(center - center[i], axis = 1) < window
        group = box[close & ~used]
        used[close] = True
        groupX = int(np.mean(group[:, 0]))
        groupY = int(np.mean(group[:, 1]))
        merged.append((groupX, groupY, window, window))

    return merged


def makeSyntheticFace(size = 24, noise = 0):

    image = np.full((size, size), 190, dtype = np.uint8)
    cropX, cropY = size // 2, size // 2
    example = size // 4

    # round face

    cv2.ellipse(image, (cropX, cropY), (size // 2 - 2, size // 2 - 1), 0, 0, 360, 205, -1)

    # eyes (dark)

    cv2.circle(image, (cropX - example, cropY - 2), max(2, size // 10), 40, -1)
    cv2.circle(image, (cropX + example, cropY - 2), max(2, size // 10), 40, -1)

    # nose

    cv2.line(image, (cropX, cropY + 1), (cropX, cropY + size // 5), 150, 1)

    if noise:

        image = image.astype(np.int16) + np.random.randint(-noise, noise + 1, image.shape)
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def makeSyntheticBackground(size = 24, seed = None):

    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 3)
    image = np.full((size, size), int(rng.integers(60, 220)), dtype = np.uint8)

    if kind == 0: # line pattern

        step = rng.integers(3, 8)

        for i in range(0, size, step):

            cv2.line(image, (0, i), (size, i), int(rng.integers(30, 230)), 1)

    elif kind == 1: # rectangle

        for _ in range(4):

            part1 = tuple(rng.integers(0, size, 2))
            part2 = tuple(rng.integers(0, size, 2))
            cv2.rectangle(image, part1, part2, int(rng.integers(30, 230)), -1)

    else: # noise

        image = rng.integers(0, 255, (size, size), dtype = np.uint8)

    return image


def buildTrainingSet(nPositive = 40, nNegative = 60, window = 24):

    rng = np.random.default_rng(0)
    positive, negative = [], []

    for i in range(nPositive):

        cropX = window // 2 + int(rng.integers(-1, 2))
        cropY = window // 2 + int(rng.integers(-1, 2))
        example = window // 4 + int(rng.integers(-1, 2))
        positive.append(makeSyntheticFace(window, noise = 8))

    for i in range(nNegative):

        negative.append(makeSyntheticBackground(window, seed = i))

    xImage = positive + negative
    y = np.array([1] * nPositive + [0] * nNegative)

    return xImage, y

def createDataset(window):

    print(f"\n[1] creating synthetic dataset...")

    image, label = buildTrainingSet(
        nPositive = 40, # default: 40
        nNegative = 60, # default: 60
        window = window
    )

    print(f"\tface image: {np.sum(label == 1)}")
    print(f"\tnon-face image: {np.sum(label == 0)}")

    return image, label


def createFeature(window):

    print(f"\n[2] build feature bank...")

    featureBank = buildFeatureBank(
        nFeature = 60,
        window = window
    )

    print(f"\tnumber of feature: {len(featureBank)}")

    return featureBank


def extractFeature(image, featureBank):

    print(f"\n[3] calculate feature...")

    x = np.zeros(
        (len(image), len(featureBank))
    )

    for i, im in enumerate(image):

        integralImage = buildIntegralImage(im)
        x[i] = extractAllFeature(
            integralImage,
            featureBank
        )

    print(f"\tfeature matrix: {x.shape}")

    return x


def trainModel(x, label):

    print(f"\n[4] train cascade..")

    cascade = trainCascade(
        x,
        label,
        stageRound = (3, 8) # default: 3, 8
    )


    print(f"\nnumber of stage: {len(cascade)}")

    return cascade


def testModel(cascade, featureBank, window):

    print(f"\n[5] test on synthetic image..")

    testImage = makeSyntheticFace(
        window,
        noise = 5
    )

    integralImage = buildIntegralImage(testImage)

    fValue = extractAllFeature(
        integralImage,
        featureBank
    )

    result = cascadeClassify(
        cascade,
        fValue
    )

    status = "face" if result == 1 else "not-face"

    print(f"\tresult: {status}")


def detectFace(cascade, featureBank, window, step):

    print(f"\n[6] detect face in real image..")

    imagePath = "PONG.png" # inputImage

    imageGray = cv2.imread(
        imagePath,
        cv2.IMREAD_GRAYSCALE
    )

    if imageGray is None:

        raise FileNotFoundError(
            f"\timage not found: {imagePath}"
        )

    imageGray = cv2.resize(
        imageGray,
        (256, 256) # resize image to (default: (256, 256))
    )

    print(f"\tloaded image: {imagePath}")

    detection = slidingWindowDetection(
        imageGray,
        cascade,
        featureBank,
        window,
        step
    )

    merged = mergeBox(
        detection,
        window
    )

    print(f"\tdetected face: {len(merged)}")

    saveResult(
        imageGray,
        merged
    )

#################################################################################

def saveResult(imageGray, box):

    print(f"\n[7] save and display result..")

    result = cv2.cvtColor(
        imageGray,
        cv2.COLOR_GRAY2BGR
    )

    for x, y, width, height in box:

        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

    outputPath = "PONG_Result.png" # outputImage

    cv2.imwrite(
        outputPath,
        result
    )

    print(f"\tsaved image: {outputPath}")

    plt.figure(figsize = (8, 8))

    plt.imshow(
        cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"viola-jone face detection - "
        f"{len(box)} face detected"
    )

    plt.axis("off")
    plt.show()


def main():

    window = 138 # default: 24
    step = 4 # default: 4

    print("=" * 70)
    print("PONG FACE DETECTION")
    print("=" * 70)

    image, label = createDataset(window)
    featureBank = createFeature(window)
    x = extractFeature(image, featureBank)
    cascade = trainModel(x, label)
    testModel(cascade, featureBank, window)
    detectFace(cascade, featureBank, window, step)

    print("\nfinished")


if __name__ == "__main__":
    main()

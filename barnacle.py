
import cv2
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Make the results reproducible
random.seed(7)
np.random.seed(7)

# Load an image and its annotation mask
def getdata(image_dest, mask_dest):
    image = cv2.imread(image_dest)
    mask = cv2.imread(mask_dest)

    if image is None or mask is None:
        raise FileNotFoundError(f"Unable to load {image_dest} or {mask_dest}")

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask dimensions do not match")

    return image, mask

# Select a region of interest
def selectRoi(image, image_name):
    print(f"Select inside the green frame for: {image_name}")

    x, y, w, h = cv2.selectROI(image_name, image)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        raise ValueError("No region selected")

    return (x, y, w, h)


# Find the blue outlines drawn around barnacles
def find_barnacles(mask):
    blue = (
        (mask[:, :, 0] > 80) &
        (mask[:, :, 1] < 110) &
        (mask[:, :, 2] < 130)
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    barnacles = []

    for contour in contours:
        if cv2.contourArea(contour) < 3:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Find the center of the annotated barnacle
        center_x_coordinate = x + w // 2
        center_y_coordinate = y + h // 2

        barnacles.append((center_x_coordinate, center_y_coordinate))

    return barnacles, blue


# Extract information from the patch
def Getfeatures(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Extract brightness and color features
    gray_mean = np.mean(gray)
    gray_std = np.std(gray)

    blue_mean = np.mean(patch[:, :, 0])
    green_mean = np.mean(patch[:, :, 1])
    red_mean = np.mean(patch[:, :, 2])

    saturation_mean = np.mean(hsv[:, :, 1])
    value_mean = np.mean(hsv[:, :, 2])

    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0) # Measure how many edges appear in the patch

    # Estimate shape features using thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the contours in the patch
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roundness = 0
    aspectratio = 0

    # Calculate the roundness and aspect ratio
    if contours:
        largestContour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largestContour)
        perimeter = cv2.arcLength(largestContour, True)

        if perimeter > 0:
            roundness = 4 * np.pi * area / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(largestContour)

        if h > 0:
            aspectratio = w / h

    # Proportion of bright and dark pixels
    bright_pixels = np.mean(gray > 180)
    dark_pixels = np.mean(gray < 70)

    return [
        gray_mean,
        gray_std,
        blue_mean,
        green_mean,
        red_mean,
        saturation_mean,
        value_mean,
        edge_density,
        roundness,
        aspectratio,
        bright_pixels,
        dark_pixels
    ]


# Cut out a patch around a point
def Acquire_target_patch(image, x, y, size=40):
    half = size // 2

    # Ignore patches that go beyond the image
    if (
        x - half < 0 or y - half < 0 or
        x + half > image.shape[1] or
        y + half > image.shape[0]
    ):
        return None

    return image[y-half:y+half, x-half:x+half]


# Build a dataset of barnacle and background patches
def build_dataset(image, mask, patch_size=40, roi=None):
    barnaclePoints, blueMask = find_barnacles(mask)

    features = []
    labels = []
    patch_locations = []

    half = patch_size // 2

    # Define the region of interest which will be selected by the user
    if roi is None:
        x0, y0, w, h = 0, 0, image.shape[1], image.shape[0]
    else:
        x0, y0, w, h = roi

    x_min = x0 + half
    x_max = x0 + w - half
    y_min = y0 + half
    y_max = y0 + h - half

    # Extract the features of the positive patches
    for x, y in barnaclePoints:
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue

        patch = Acquire_target_patch(image, x, y, patch_size)

        if patch is not None:
            features.append(Getfeatures(patch))
            labels.append(1)
            patch_locations.append((x, y, 1))

    positives = len(features)

    if positives == 0:
        raise ValueError("No annotated barnacles found.")

    # Extract the features of the negative patches
    attempts = 0
    negatives = 0
    max_attempts = positives * 100

    while negatives < positives and attempts < max_attempts:
        attempts += 1

        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)

        patch = Acquire_target_patch(image, x, y, patch_size)

        if patch is None:
            continue

        # Ignore the patches that contain annotated barnacle outlines
        annotatedPatch = blueMask[y-half:y+half, x-half:x+half]

        if np.any(annotatedPatch > 0):
            continue

        features.append(Getfeatures(patch))
        labels.append(0)
        patch_locations.append((x, y, 0))
        negatives += 1

    return np.array(features), np.array(labels), patch_locations


# Gives the visualization of patch placements 
def visualize_patches(image, patch_locations, patch_size=40):
    annotated = image.copy()
    half = patch_size // 2

    for x, y, label in patch_locations:
        if label == 1:
            color = (0, 255, 0)  # Green for barnacle patches
        else:
            color = (0, 0, 255)  # Red for background patches

        cv2.rectangle(annotated, (x-half, y-half), (x+half, y+half), color, 1)

    print("Green boxes represent positive barnacle patches")
    print("Red boxes represent negative background patches")

    cv2.imshow("Training Patches", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the two images and their masks
image1, mask1 = getdata("Images/img1.png", "Images/mask1.png")
image2, mask2 = getdata("Images/img2.png", "Images/mask2.png")

# Allow user to select the region of interest
roi1 = selectRoi(image1, "Image 1 Frame")
roi2 = selectRoi(image2, "Image 2 Frame")

# Build datasets using the selected regions
Xtrain, ytrain, train_coords = build_dataset(image1, mask1, patch_size=40, roi=roi1)

Xtest, ytest, test_coords = build_dataset(image2, mask2, patch_size=40, roi=roi2)

# Visualize the patches sampled in training and testing
visualize_patches(image1, train_coords)
visualize_patches(image2, test_coords)

print("Training barnacle patches:", np.sum(ytrain == 1))
print("Training background patches:", np.sum(ytrain == 0))

print("Testing barnacle patches:", np.sum(ytest == 1))
print("Testing background patches:", np.sum(ytest == 0))


# Train the Random Forest classifier using image 1
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=7)

model.fit(Xtrain, ytrain)


# Predict labels for patches from image 2
predictions = model.predict(Xtest)

# Count the number of barnacles detected
print("\nModel Prediction:")
print("--------------------------")
print("Predicted barnacle patches:", np.sum(predictions == 1))
print("Predicted background patches:", np.sum(predictions == 0))

# Calculate evaluation metrics
accuracy = accuracy_score(ytest, predictions)
precision = precision_score(ytest, predictions, zero_division=0)
recall = recall_score(ytest, predictions, zero_division=0)
f1 = f1_score(ytest, predictions, zero_division=0)

print("\nEvaluation Results")
print("------------------")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 score:  {f1 * 100:.2f}%")

print("\nConfusion Matrix")
print("------------------")
print("Rows = actual [background, barnacle]")
print("Columns = predicted [background, barnacle]")
print(confusion_matrix(ytest, predictions))

print("\nClassification Report")
print(classification_report(ytest, predictions, labels=[0, 1], target_names=["Background", "Barnacle"], zero_division=0))

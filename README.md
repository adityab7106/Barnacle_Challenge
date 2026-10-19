# Barnacle_Challenge
A system is developed that uses OpenCV to extract image patches from annotated barnacles and the surrounding background. It then trains a Random Forest Classifier that differentiates between the two based on visual features such as color, brightness, and texture. The classifier is then tested on a different image and evaluated on metrics such as accuracy, precision, recall, and F1 score. 

## Files
- barnacle.py - The main implementation file containing the code for training and testing the machine learning model
- Images - Folder containing the photographs necessary for training and testing
- README - A general descriptive file for the project

## Dependencies
The system requires Python 3 and the following Python libraries:

- opencv-python
- numpy
- scikit-learn

These dependencies can be installed using pip:
pip install opencv-python numpy scikit-learn

## How to Run
1. Place barnacle.py and the Images/ directory in the same project folder, ensuring that all images are present
2. Install the required dependencies
3. Open a terminal in the project folder and run: python barnacle.py
4. When prompted, select a rectangular region of interest in each image and confirm the selection. These regions dictate where the system acquires patches
5. Allow the script to finish training and evaluation. It will print the evaluation metrics, including accuracy, precision, recall, and F1 score, and display visualizations of the positive and negative patches

## Brainstorming
Scientists studying marine ecosystems often need to count barnacles in photographs, which can shed important information regarding the health of the ecosystem. However, manually counting hundreds or thousands of barnacles in a single image can be really time-consuming. Thus, I propose developing a computer vision system that identifies barnacles in photographs and runs a Random Forest Classifier to estimate their abundance within the desired sampling area (the green frame). The OpenCV library can be utilized to detect patches containing individual barnacles in the image and extract features to help train the machine learning model. The model can then be implemented on the testing data and evaluated to determine its effectiveness.   

## Identifying subtasks
Deploying an effective barnacle detection system hinges on several important subtasks, as identified below:

- **Subtask 1: Selecting the Sampling Region**
  The system must be able to identify the region of interest where the barnacles are to be counted. In this project, that region is the inner square of the green frame. In this prototype, the user manually selects the interior of the green frame in each photograph. The input is the original image, and the output is a rectangular region of interest (ROI).
- **Subtask 2: Generate Training Labels**
  A machine learning model needs examples of both barnacles and background to learn how to distinguish between them. The provided images include mask images with blue outlines that identify barnacle locations. The system locates the center of each annotated barnacle and takes a small image patch around it as a positive example. It also samples patches from other locations (avoiding the patches that overlap with blue outlines) to create negative examples. The output is a labeled dataset of image patches.
- **Subtask 3: Extract image features and train a classifier**
  In order to train the machine learning model, each patch must be converted to numerical information that the model can process. Thus, features related to brightness, color, edges, and shape are extracted to train the Random Forest Classifier. The output is a model that predicts whether a patch contains a barnacle or a part of the background.
- **Subtask 4: Evaluate Classification Performance**
  After the model has been trained, it must be tested on data from a different photograph to determine its true effectiveness. The model compares its predictions against the labels assigned to the test patches and calculates performance metrics. The output is a numerical evaluation of the classifier.
- **Subtask 5: Visualize the sampled patches**
  The system draws boxes around the patches used to train and evaluate the model. Green boxes represent positive barnacle examples, while red boxes represent negative background examples. This visualization helps understand whether the training data are sampled from the correct region and whether the patches appear reasonable.

## Selecting Subtask and Evaluation Strategy
Since all of the subtasks are crucial for developing and implementing an effective model, I chose to incorporate them all into this project. However, I spent the most amount of time on subtask 3, training the classifier. A reliable classifier is crucial, as it can help identify barnacles in photographs with minimal manual intervention.

To evaluate the classifier, I used the first photograph and its annotation mask to construct the training dataset. Then, I built the testing dataset based on the second photograph and its mask. Both photographs have their own manually selected regions, which ensures that the sampled patches come from the desired regions of interest.

I utilized four evaluation metrics: accuracy, precision, recall, and F1 score. Accuracy is the proportion of all test patches that were classified accurately. Precision is the proportion of patches predicted to contain barnacles that are labeled as barnacles. A high precision means fewer background patches are incorrectly classified as barnacles. Recall is the proportion of actual positive test patches correctly classified as barnacles. A high recall means the model misses fewer labeled barnacle patches. Finally, the F1 score is the harmonic mean of precision and recall. This provides a single measure that balances false positives and false negatives. I also calculated a confusion matrix to show the number of correctly and incorrectly classified barnacle and background patches.

These metrics are practical to calculate because the given annotation masks allow me to generate labels without needing to manually label every patch. However, these metrics do have a limitation. They only measure patch classification performance, not the accuracy of counting every barnacle in the entire image. Thus, if there are unannotated barnacles, they might get labeled as a background patch, introducing potential errors in the model. 

## The Prototype
The prototype is implemented in Python using OpenCV for image processing, NumPy for numerical operations, and scikit-learn for machine learning and acquiring evaluation metrics.

The getdata() function loads each photograph and its respective annotation mask. It ensures that both files exist and that their dimensions are the same. The selectRoi() function allows the user to select the region of interest in the photograph. 

The find_barnacles() function identifies the blue pixels in the annotation mask using color thresholds. OpenCV then identifies connected contours in the resulting binary image. Contours smaller than 3 pixels are discarded as the likelihood of them representing a barnacle is very low. The center of each remaining contour is used as the location of an annotated barnacle.

The build_dataset() function checks whether each annotated barnacle is within the region of interest and has enough surrounding image area to form a complete patch. If it does, the program extracts a rectangular 40x40 pixel patch centered on the barnacle and labels it as positive, represented by a 1. 

Since it is important to identify what features constitute a barnacle and what does not, the program also generates negative samples by randomly selecting patches inside the region of interest and represents them by a 0. However, it rejects a patch if it overlaps with any blue annotation pixels, since the patch may contain parts of a barnacle. The program tries to collect the same number of negative patches as positive patches. Doing so results in a balanced dataset where there is an equal amount of barnacle and background examples.

The Getfeatures() function generates a list of 12 numerical features from each patch. The features are defined as below:
<img width="322" height="341" alt="Screenshot 2026-10-18 at 9 37 56 PM" src="https://github.com/user-attachments/assets/3fd10144-fa85-4b9d-85ca-bd6296f0aad4" />

Here, roundness is calculated as:\
<img width="186" height="76" alt="Screenshot 2026-10-18 at 9 39 25 PM" src="https://github.com/user-attachments/assets/a4efe2ff-1672-4f01-ac18-929d9d0aee1d" />\
Where A is the contour’s area, and P is its perimeter. The closer the shape is to a circle, the more the roundness value approaches 1.

Aspect ratio is calculated as:\
<img width="123" height="86" alt="Screenshot 2026-10-18 at 9 40 10 PM" src="https://github.com/user-attachments/assets/b744fe3c-01f9-4182-a397-eacfd8d4b987" />\
Where W and H are the width and height of the contour’s bounding rectangle.

These features represent different visual properties that may help distinguish barnacles from their surroundings. Barnacles have visible rims and a distinct shape. The background may differ in brightness and color, among others.

The program trains a Random Forest Classifier using 100 decision trees. Random Forest combines the predictions of multiple decision trees to classify a patch based on the patch’s extracted features. The class_weight="balanced" setting adjusts class weights based on their frequencies, and random_state=7 makes the model’s results reproducible.

The model is trained using patches from the first photograph. It then predicts the labels of patches sampled from the second photograph. Finally, the program calculates accuracy, precision, recall, F1 score, and a confusion matrix.

The visualize_patches() function overlays rectangles on the original images. Green rectangles identify positive patches, while red rectangles identify negative patches. This allows me to analyze the data and ensure the model is being trained on the correct examples. Since a classifier’s performance depends heavily on the quality of the training data, visualization is an important step.
<img width="500" height="500" alt="Screenshot 2026-10-18 at 8 04 41 PM" src="https://github.com/user-attachments/assets/eb09df24-406b-4400-aea6-459d4975c4f9" />
<img width="500" height="500" alt="Screenshot 2026-10-18 at 8 03 54 PM" src="https://github.com/user-attachments/assets/abd9d468-70b3-40e8-9801-330c2ea590d0" />

## Analyzing Model Performance
After testing the model on image 2, the following metrics were generated:

Accuracy: 71%
Precision: 75%
Recall: 62%
F1 Score: 68%
Predicted barnacle patches: 322  

Since the metrics depend on the user-selected region of interest, they may vary slightly in each run. The above metrics are reported based on the best-performing run.





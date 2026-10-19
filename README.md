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




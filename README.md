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


# Spam-Detection-Using-Python

## Description 
The Spam Message Detector is a Machine Learning project that classifies text messages as Spam or Normal (Ham). It uses the Multinomial Naive Bayes algorithm with TF-IDF Vectorization to analyze message content and predict whether a message is spam. The project includes a simple and user-friendly Tkinter GUI, allowing users to enter a message, detect spam, view the prediction confidence, and see the model accuracy.

##  Features
- Detects whether a message is Spam or Normal (Ham).
- Uses Multinomial Naive Bayes for text classification.
- Converts text into numerical features using TF-IDF Vectorizer.
- Displays the prediction result instantly.
- Shows spam probability percentage.
- Displays the trained model accuracy.
- User-friendly GUI built with Tkinter.
- Clear button to reset the input and output.
- Handles empty input with warning messages.
- Automatically loads and preprocesses the dataset.

## Technologies Used
- Python 
- Tkinter 
- Pandas 
- Scikit-learn 
- TF-IDF Vectorizer 
- Multinomial Naive Bayes 
- Train-Test Split 
- Accuracy Score 
- CSV Dataset 

 ## Dataset
 - spam.csv

## Project Structure
```
Spam-Message-Detector/
│── spam.csv
│── spam_detection.py

```

##  Installation

### Install required libraries

```bash
pip install pandas scikit-learn
```

*(Tkinter is included with most Python installations.)*

##  Run the Project

```
python spam_detection.py
```

## 🔄 Working Process

1. Load the **spam.csv** dataset.
2. Clean and preprocess the data.
3. Convert text into TF-IDF feature vectors.
4. Split the dataset into training and testing sets.
5. Train the Multinomial Naive Bayes model.
6. Evaluate model accuracy.
7. Enter a message in the GUI.
8. Click **Detect**.
9. View the prediction and spam probability.



## Sample Output

### Example 1

**Input**

```
Congratulations! You have won ₹50,000. Click here to claim.
```

**Output**

```
⚠ SPAM MESSAGE
Spam Probability: 98.45%
```

### Example 2

**Input**

```
Hi, are we meeting at 5 PM today?
```

**Output**

```
✓ NORMAL MESSAGE
Spam Probability: 2.87%
```









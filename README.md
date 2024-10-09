# Predicting Equipment Failure using Logistic Regression and Random Forest

![Designer (8)](https://github.com/user-attachments/assets/131b95e4-95c4-4fc6-95d4-5d5309c5f7fc)


## Table of Contents
- Description
- Inspiration
- Source of Dataset
- Interesting Facts from Exploratory  Data Analysis
- Get Started
 - Pre-installation
 - Set-up
- Report

- Result
- Challenges
- Contributors
## Description
This project focuses on predicting equipment failure by leveraging feature engineering and machine learning models, specifically Logistic Regression and RandomForestClassifier. By creating time-based features and applying rolling statistics, we aim to enhance model accuracy in predicting whether an event (e.g., click) will result in attribution.

The model achieved 100% accuracy on the test set, which showcases the importance of effective feature engineering and proper model selection in predictive analytics.



## Inspiration
The inspiration for this project comes from real-world problems involving predictive maintenance. Equipment failures are costly and unpredictable. The goal is to create a system that can forecast equipment failures before they occur, saving operational downtime and resources.

## Source of Dataset
[Dataset Link](https://www.kaggle.com/datasets/matleonard/feature-engineering-data/data)
## Interesting Facts from Exploratory Data Analysis
- Click-time patterns: Most events tend to happen during specific hours of the day, with peak activity in the afternoon.
- Attribution rates: There is a clear pattern of attribution related to certain channels, suggesting that some channels have higher success rates.
- Rolling statistics: The rolling count of clicks for specific IPs highlighted trends where certain users generate frequent activities before a successful attribution occurs.g.


###  Get Started
- Pre-installation
Ensure you have the following dependencies installed:

- Python 3.7+
- Required libraries: pandas, numpy, scikit-learn, seaborn, matplotlib
You can install the required dependencies using pip:
```bash 
pip install pandas numpy scikit-learn seaborn matplotlib



```

- Set-up
 1. Clone this repository:
```bash
git clone https://github.com/yourusername/equipment-failure-prediction.git


```
2. Navigate to the project directory:
```bash
cd equipment-failure-prediction
```
3. Run the Jupyter notebook or Python script to generate features and train the models:
```bash
python feature_engineering_and_modeling.py


```
## Report

[Report.pdf](https://github.com/user-attachments/files/17306320/The.report.on.Predicting.Equipment.Failure.using.Logistic.Regression.and.Random.Forest.Project.pdf)
## Results
After feature engineering and training the models:

- Logistic Regression Accuracy: 100%
- Random Forest Accuracy: 100%
Both models performed exceptionally well due to the strength of the engineered features, which captured the underlying patterns in the data.


## Challenges
- Data Imbalance: One challenge was the potential imbalance in the is_attributed target variable, which could skew model predictions. We addressed this by analyzing feature distributions and ensuring proper train-test splitting.

- Overfitting: The high accuracy of the model could be a sign of overfitting. We plan to investigate this further by using techniques like cross-validation and adjusting hyperparameters.


## Contributing
- Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Developed by [MD Rashidul Islam](https://github.com/mrirashid/)


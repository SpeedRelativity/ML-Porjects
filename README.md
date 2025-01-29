Housing Price Prediction Using Machine Learning
This project aims to predict housing prices based on features like square footage, number of bedrooms, and number of bathrooms. It showcases data cleaning, visualization, and machine learning techniques using real estate data scraped from Realtor.com.

Project Overview
Objective: Build an end-to-end machine learning pipeline to predict housing prices.
Key Features:
Data cleaning and preprocessing.
Exploratory data analysis with visualizations.
Machine learning model implementation and evaluation.
Tools & Libraries: Python, Pandas, Matplotlib, Scikit-Learn.
Dataset
The dataset contains information about houses, including:

Price
Square Footage
Number of Bedrooms
Number of Bathrooms
Data Preprocessing Steps
Removed unnecessary columns.
Cleaned formatting (e.g., removed $ and , from numerical data).
Handled missing values by dropping rows with null data.
Converted all columns to numeric for analysis.
Exploratory Data Analysis
Key Insights:
Visualized relationships between:
Square Footage vs. Price
Bedrooms vs. Price
Bathrooms vs. Price
Scatter plots reveal trends that help guide feature selection for modeling.


Machine Learning Pipeline
Model Used: Linear Regression.
Features:
Bedrooms
Bathrooms
Square Footage
Model Performance:
Mean Squared Error (MSE): Your MSE Value
R² Score: Your R² Value
Steps:
Split data into training and testing sets (80%/20%).
Trained a linear regression model using Scikit-Learn.
Evaluated the model on the testing set with performance metrics.


# Kaggle_Rossmann
## I.Definition
## Project Overview
Rossmann is Europe's largest pharmaceutical company with approximately 3,000 drugs sold in stores in 7 different countries. In order to be able to create an effective and reasonable employee timetable to increase productivity and motivation, current Rossmann managers require the use of skills such as machine learning to create a model for forecasting daily sales over the next six weeks. Daily sales will be affected by a number of factors, such as promotions, competitors, holidays, seasonal factors, etc. The data entered has training and test sets, as well as store supplemental information. In summary, this problem is a predictive problem, which is what machine learning is good at. After providing enough data (more than 1 million), machine learning can more accurately create a model based on this data. Identify attributes that have a greater impact on price factors to simulate forecasts for future prices.
## Problem Statement
This project is based on real data to predict future sales, and has given the impact factor in the data set to determine that this is a supervised regression problem. I will preprocess the data based on historical sales information and influencing factors (ie training data and store type), such as processing default values or outliers, dividing the training set, and operating on categorical data and numerical data, to choose important features or incorporate new features. Next, train the XGBoost models to adjust weights based on the validation data. Either select a single model or ensemble models.
## Metrics
Because it was submitted to kaggle for evaluation, it was evaluated using the rmspe method. RMSPE is very sensitive to very large or very small errors in a set of predictions, so it is a good reflection of the precision of model predictions. For the sales volume that needs to be predicted in the project, RMSPE can well represent the effect of the model. Its formula is:
![image](https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/rmspe.png)
Where y_i represents the sales of a single-day store and yhat_i represents the corresponding forecast. Ignoring the selling is 0 in the rating.
## II. Analysis
## Data Exploration

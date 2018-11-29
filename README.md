# Kaggle_Rossmann
## I.Definition
## Project Overview
Rossmann is Europe's largest pharmaceutical company with approximately 3,000 drugs sold in stores in 7 different countries. In order to be able to create an effective and reasonable employee timetable to increase productivity and motivation, current Rossmann managers require the use of skills such as machine learning to create a model for forecasting daily sales over the next six weeks. Daily sales will be affected by a number of factors, such as promotions, competitors, holidays, seasonal factors, etc. The data entered has training and test sets, as well as store supplemental information. In summary, this problem is a predictive problem, which is what machine learning is good at. After providing enough data (more than 1 million), machine learning can more accurately create a model based on this data. Identify attributes that have a greater impact on price factors to simulate forecasts for future prices.
## Problem Statement
This project is based on real data to predict future sales, and has given the impact factor in the data set to determine that this is a supervised regression problem. I will preprocess the data based on historical sales information and influencing factors (ie training data and store type), such as processing default values or outliers, dividing the training set, and operating on categorical data and numerical data, to choose important features or incorporate new features. Next, train the XGBoost models to adjust weights based on the validation data. Either select a single model or ensemble models.
## Metrics
Because it was submitted to kaggle for evaluation, it was evaluated using the rmspe method. RMSPE is very sensitive to very large or very small errors in a set of predictions, so it is a good reflection of the precision of model predictions. For the sales volume that needs to be predicted in the project, RMSPE can well represent the effect of the model. Its formula is:  
<img align="middle" width="600" height="250" src=https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/rmspe.png>
<br>Where y_i represents the sales of a single-day store and yhat_i represents the corresponding forecast. Ignoring the selling is 0 in the rating.
## II. Analysis
## Data Exploration
The data is downloaded from Kaggle and has three files, store.csv, train.csv and test.csv. The training set (more than 1.01 million pieces of data) and the test set (about 40,000 pieces of data) contain characteristics such as daily turnover, number of customers, business, promotion, and holidays. Store datasets include store type, classification level, competitive information, and promotional information.    
Whether the characteristics of the training set are open, whether it is promoted, holidays, etc. are categ ry data (ie 0 or 1); the number of customers, the competition information is u erica data. And these data are also the main features of training. The daily turnover will be the predicted target. Data fields:
 * Id - an Id that represents a (Store, Date) duple within the test set
 * Store - a unique Id for each store
 * Sales - the turnover for any given day (this is what you are predicting)
 * Customers - the number of customers on a given day
 * Open - an indicator for whether the store was open: 0 = closed, 1 = open
 * StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
 * SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
 * StoreType - differentiates between 4 different store models: a, b, c, d
 * Assortment - describes an assortment level: a = basic, b = extra, c = extended
 * CompetitionDistance - distance in meters to the nearest competitor store
 * CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
 * Promo - indicates whether a store is running a promo on that day
 * Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
 * Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
 * PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

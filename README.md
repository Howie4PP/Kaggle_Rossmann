# Kaggle_Rossmann
## I.Definition
## Project Overview
Rossmann is Europe's largest pharmaceutical company with approximately 3,000 drugs sold in stores in 7 different countries. In order to be able to create an effective and reasonable employee timetable to increase productivity and motivation, current Rossmann managers require the use of skills such as machine learning to create a model for forecasting daily sales over the next six weeks. Daily sales will be affected by a number of factors, such as promotions, competitors, holidays, seasonal factors, etc. The data entered has training and test sets, as well as store supplemental information. In summary, this problem is a predictive problem, which is what machine learning is good at. After providing enough data (more than 1 million), machine learning can more accurately create a model based on this data. Identify attributes that have a greater impact on price factors to simulate forecasts for future prices.
## Problem Statement
This project is based on real data to predict future sales, and has given the impact factor in the data set to determine that this is a supervised regression problem. I will preprocess the data based on historical sales information and influencing factors (ie training data and store type), such as processing default values or outliers, dividing the training set, and operating on categorical data and numerical data, to choose important features or incorporate new features. Next, train the XGBoost models to adjust weights based on the validation data. Either select a single model or ensemble models.
## Metrics
Because it was submitted to kaggle for evaluation, it was evaluated using the rmspe method. RMSPE is very sensitive to very large or very small errors in a set of predictions, so it is a good reflection of the precision of model predictions. For the sales volume that needs to be predicted in the project, RMSPE can well represent the effect of the model. Its formula is:  
<img width="600" height="250" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/rmspe.png">
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
<p>In such a large data set, there should be default values or outliers. For the default value, if it occupies a small proportion in whole data set, which coule be deleted or filled it with the average value. For example, the record with the sales amount of 0 is not meaningful, this experiment only Data with a store is 'Open' and 'Sales' greater than 0 is used. If the default value occupys a large percentage, such as the entire property is mostly empty, which could be considered to deprecate this property. On the other hand, for outliers, because this problem is a regression problem, its quite sensitive with outliers. Visualization (such as quartile or variance processing) can be used to confirm that outliers are not an important factor (such as the data of two bosses in Enron cases appear as outliers, but cannot be removed), if not important, the data is deleted directly.</p>

## Exploratory Visualization
After preliminary exploration, it was found that Custers and Sales had a lot of outlier values.
<img  width="400" height="450" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/1.png">
<br>This is a missing feature of several features such as Customers, Open, Promo, etc. It can be seen that the missing conditions of Customers, Open and Sales are basically the same, so the missing values can be discarded by simple filtering during preprocessing.
<img width="700" height="280" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/2.png">
<br>This is a timeline graph of the average sales volume for each day of the week. As you can see from the chart, at the weekend, sales volume reached its highest level and then began to decline. Saturday was the least average day for guests. When the traffic volume is the highest on Sunday, the sales volume is not high. Monday is the opposite.
<img width="700" height="300" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/3.png">
<br>This is a line chart of the average sales volume for each month from 2013 to 2015. The peak of sales is around Christmas, and there is a wave of sales after Christmas.
<img height="300" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/4.png">
<br>Correlation analysis of individual features with the ‘Sales’ tag can help determine which features are not very relevant to avoid overfitting and resource wastage during training. The following is an unprocessed feature correlation visualization.     
<img height="680" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/5.png">
<br>This is the visualization after processing.
<img height="680" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/6.png">
## Algorithms and Techniques
From the visualization, it can be seen that sales are closely related to time, but there are many other characteristics that affect it, such as the distance of competitors, whether they are in school holidays, and whether there is advertising promotion. Moreover, the missing values and outliers in the data will affect the forecast of sales. In order to achieve the best prediction, the abnormal data of Sales and Customers should discard.

<br>The second step is data preprocessing and feature processing and transformation, including processing missing values and making appropriate changes to important features or generating partial new features from old features to facilitate training of the model. For example, convert the characteristics of different characters into numbers, such as ‘mappings = {'0': 0, 'a':1, 'b':2, 'c':3, 'd':4}’. Due to the close relationship with time characteristics, time features are split and transformed, and ‘weekOfYear’, 'CompetitionOpen' and 'PromoOpen' are added as new features. Convert the 'PromoInterval' to 'IsPromoMonth', indicating whether a store is in a promotion month.
<br>In the third step, I tried to use the cross_validation.train_test_split to randomly divide the training data into a training set and a test set. Then I started modelled, tested, and submiited it to Kaggle to test, but I could not reach the target of 0.11. Thus, I tried to use another method to process the data, which is only train the XGBoost model.
 * XGBoost: This is a model for supervised learning. The excellent performance of Gradient Boosting and the efficient implementation of XGBoost make it perform well in this project, and the first place in the Kaggle competition is mainly using this algorithm. Forecasting the results. XGBoost customizes a data matrix class, DMatrix, which is pre-processed at the beginning of the training to improve the efficiency of each iteration. Under supervised learning, the general linear model uses a given training set (including multidimensional features Xi) to predict the target variable Yi, which is:
 <img width="300" height="100" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/7.png">
The predicted values have different interpretations depending on whether the task is regression or classification. And the parameters are uncertain and need to be learned from the data. In linear regression, the parameter refers to the coefficient θ.
In general, the objective function consists of two parts: the training set loss and the regularization term.
<img width="300" height="100"src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/8.png">
The Logistic loss is calculated according to the following formula:
<img width="300" height="100" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/9.png">
Regularization controls the complexity of the model and avoids overfitting. Based on the above formula, the derived mathematical formula of the XGBoost model is:
<img width="300" height="100" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/10.png">
The principle of XGBoost is to predict the score of each tree, increase the weight of the training error (lower score), and then invest in the next training, so that the next training is easier to identify the wrong classification. example. These weak classifiers are finally weighted and added. In this training, according to the weight used to update the adjustment, there is the following formula:
<img width="300" height="100" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/11.png">
among them:
<img width="300" height="100" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/12.png">
The main parameters of the XGBoost model that need to be debugged are as follows:

  * Eta: Shtinkage parameter, used to update the weight of the child node, multiplied by the coefficient, equivalent to the learning rate
  * nthread: number of threads
  * subsample: Random sample. Lower values make the algorithm more conservative and prevent overfitting, but too small values can cause under-fitting
  * colsample_bytree: Column sampling, which performs column sampling on the features used to generate each tree. The general setting is: 0.5-1
## Benchmark
The project uses RMSPE as the evaluation index, and the lower the score, the better the model performance. According to the data on Kaggle, I guess the pass value is about 0.2. But I will set the benchmark at 0.11773 and try to get into the top 10% of the private leaderboard.
## III.Methodology
## Data Preprocessing
The first is to fill the missing values with 0 or 1. For example, the open data in the test data is filled with 1, and the missing data in the store is mostly related to competitors and promotions. The test finds that the effect of filling 1 is not good, so Use 0. After that, the time characteristics are classified in detail as described above, mainly by dividing the date including the year, month, and day into Day, Month, Year, DayOfWeek, WeekOfYear and other features. At the same time, the competitor information and promotion information contained in the store information have also been processed here, adding two new features:
* CompetitionOpen is calculated according to 12 * (Year - CompetitionOpenSinceYear) + (Month - CompetitionOpenSinceMonth), and filters out data greater than 0. If it is not greater than 0, it is 0.
* PromoOpen is calculated according to 12 * (Year - Promo2SinceYear) + (WeekOfYear - Promo2SinceWeek) / 4.0, and filters out data greater than 0. If it is not greater than 0, it is 0.
The purpose is to calculate the business hours of a store’s competitors and the time the store has been promoted.Then after exploring through visualization, I delete the outliers and features that are not highly correlated. To avoid training and predictions that affect the regression model.    
The continuous features are normalized according to the numerical distribution, such as the Sales feature, I will perform logarithmic scaling, and the results are visualized as follows:       
<img width="700" height="450" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/13.png">
One-hot coding is performed on discrete features.      

## Implementation
Testing the last 6 weeks of data as a hold_out data set with the following code:
Train = train.sort_values(['Date'],ascending = False)
Ho_test = train[:6*7*1115],ho_train = train[6*7*1115:]
Only data with ‘open’ is not 0 and sales greater than 0 is taken. The feature is then split with the tag, and the tag is logarithmically processed to make the data distribution normal, to avoid skewing the data distribution.
The training parameters of the XGBoost model are as follows:      
<img width="500" height="300" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/14.png">
<br>After the training is completed, the test set that was previously retained is used for detection and the prediction results of the data set are retained.    
## Refinement
When the prediction has completed, the results need to be analyzed. Only the 10 predictions with the largest deviation are analyzed here. As can be seen from the test results below, the best weight is 0.996 and the score is 0.120463.
<img width="500" height="400" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/15.png">     
<br>However, after considering the generalization of the model, the next adjustment is to correct the model with a correction factor of 0.990. code show as below:    
<img width="600" height="400" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/16.png">     
<br>After the correction, multiple calibration models and initial models are merged and retrained.
Through the query of the data, the weighted fusion is used here, the reason is that the effect is better than the simple average fusion according to the query of the data.    
## IV. Results
## Model Evaluation and Validation
The following is the score for the initial model and the fused modified model:
<img height="300" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/17.png">   
<br>It can be seen that the score of the model that was originally trained has been higher, but the fusion model has improved the score by 0.01 points and more, and the performance is more excellent. The public score is greater than the private score, which also shows that the performance is reasonable. Trustworthy, and the generalization ability of the fusion model is better than the original model.
## Justification
Compared with the comprehensive decision tree model, the XGBoost model has the biggest advantage of greatly shortening the time. After the adjustment, the trained models have laid the foundation for integration because of their independence. After the optimization, the scores have been greatly improved, which indicates that the fusion model is reasonable.
## V. Conclusion
## Free-Form Visualization
<img width="600" height="380" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/18.png">     
<br>It shows that XGBoost is sensitive to time from the figure. This learner is mainly learned from the time rule of Rossmann sales, and then the characteristics of each store. It can be said that the XGBoost model has a more accurate prediction for each store with different characteristics. This also shows that the XGBoost model is more reliable for predicting time-based data, and the effect of correcting the fusion is better.      
After obtaining the final model, the forecast and real sales comparisons were made for any of the three stores. The visualization of one of the stores is as follows:
<img height="400" src="https://github.com/Howie4PP/Kaggle_Rossmann/blob/master/images/19.png"> 
<br>It can be seen that the predictions displayed by the model are roughly similar to the actual sales. It also shows from the side that the model training is better.
## Reflection
In fact, this project was done twice, and it took a lot of time to train. The first time was based on previous experience using the decision tree model and XGBoost fusion, and the prediction was made. The final score was around 0.15, no matter how many parameters were adjusted. The scores have always been fixed in this range, and it has not been able to reach the previously set goals. Later, after exploration and inquiry, it was found that the XGBoost model can be used for training alone, and the model after correction and adjustment is better. Of course, when doing this training, the confusion is the selection, clean-up and transformation of the project features. Several important features have been deleted, sometimes it is not known which is an important feature; sometimes I don’t know what visual map to use. Intuitively express the effect I want to present; when adjusting, the correction coefficient can not be determined, do not know which weight is the most appropriate; the model fusion method is mean fusion or weighted fusion. Fortunately, the final step is completed and the target is achieved.   

## Improvement
Although the final score has reached the preset goal, there is still a lot of room for improvement, such as:    
   * Pre-processing: For the processing of missing values and outliers, in the project, the outliers are directly discarded. If you can pre-train a model, it may be better to re-predict the missing and outliers. Outliers may be more reasonable based on variance-based choices.
   * Feature selection: A good feature segmentation or feature transformation can greatly improve the prediction results. I think it can improve the feature selection and processing.
   * Model selection: Only the XGBoost model is used here, maybe you can try some linear models for Ensemble.
   * Try to use the Stacking.
   
## Reference
<a href="https://www.kaggle.com/tqchen/understanding-xgboost-model-on-otto-data">理解在Otto数据库中的XGBoost模型</a>

<a href="https://www.kaggle.com/c/rossmann-store-sales/leaderboard">Rossmann-kaggle项目</a>

<a href="https://cn.udacity.com/">监督回归学习</a>
  
<a href="https://zhuanlan.zhihu.com/p/25836678">【机器学习】模型融合方法概述</a>
  
<a href="https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/">如何在 Kaggle 首战中进入前 10%</a>

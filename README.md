# Loan_Application_Screener

## Background
LendingClub is a peer-to-peer lending company in which private investors can invest their own money by lending it to other individuals. Investors put in a sum of money and then through the LendingClub platform, distribute their funds across multiple loans with different credit risks. Investors earn a portion of the interest charged to the borrowers - with the potential to earn much higher interest rates overall than paid for deposit funds at banks, credit unions, or money markets. However, the risk of borrower default can be much higher as well. To aid the Investor, each potential borrower is evaluated for their overall risk of "paying back the loan" by LendingClub. Lending Club makes millions of individual loans and keeps original loan application files and if the borrower ultimately paid back the loan or defaulted. 

This data is stripped of borrower ID information making it anonymous, but the variables during application are kept in tact so they can be evaluated by a machine learning model. To facilitate building and testing various ML Models, a CSV file was provided for "train" and "test" data. 

## Challenges
Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, we are employing different techniques to train and evaluate models with unbalanced classes, then comparing their precision, recall, and overall F1 scores to see if any of the models standout as superior to the others, making a recommendation on possible use.  

## Resources: 
* 3.6 Python plus Numpy library
* Imbalanced-learn and scikit-learn libraries 

## Data Proprocessing
The loanstats_2019Q1.csv file - when read into a pandas dataframe contained 86 columns of data -- some relevant to the loan process and some not. The data was wranggled with the following changes:
* Null values dropped
* Null rows dropped
* Interest Rate column converted to numerical values
* The target column "loan_status" categorical values changed. 
* Using the pd.get_dummies method, categorical data changed to numercial encoded data with additional columns added to hold the values. 
* The target "loan_status' column was then dropped from the "features" list in "X" and the target "y" was defined as "loan_status" and converted from a df to a series. 
* The data was then split using the "train_test_split method producing:
- low_risk loan applications = 68470 records
- high_risk loan applications = 347 records

Clearly, this is an imbalanced dataset therefore sampling must be done with care using Oversampling, Undersampling, and Combination to see if a model could be produced which produced accurate results. 

For each of 4 Models, the following process was used:
* View the count of the target classes using Counter from the collections library.
* Use the resampled data to train a logistic regression model.
* Calculate the balanced accuracy score from sklearn.metrics.
* Print the confusion matrix from sklearn.metrics.
* Generate a classication report using the imbalanced_classification_report from imbalanced-learn.

The Balanced Accuracy, Confusion Matrix, and Classification Report was then compared acrross the 4 different models looking for differentiation on strengths and weaknesses of each model. 

## Model 1 - Random Oversampling
* Resampling Produced - 51366 for both "low_risk" and "high_risk" target categories. 
* Balanced_Accuracy = 0.6603423204808787
* Confusion Matrix: 
* array([[  75,   26],
*        [7216, 9888]
* Classificaton_report:
*                   pre       rec       spe        f1       geo       iba       sup
*   high_risk       0.01      0.74      0.58      0.02      0.66      0.44       101
*    low_risk       1.00      0.58      0.74      0.73      0.66      0.42     17104
* avg / total       0.99      0.58      0.74      0.73      0.66      0.42     17205

## Model 2 - SMOTE Oversampling 
* Resampling Produced - 51366 for both "low_risk" and "high_risk" target categories. 
* Balanced_Accuracy = 0.6537310478007576
* Confusion Matrix: 
* array([[   63,    38],
*        [ 5410, 11694]
* Classification_report:
*                   pre       rec       spe        f1       geo       iba       sup
*  high_risk       0.01      0.62      0.68      0.02      0.65      0.42       101
*   low_risk       1.00      0.68      0.62      0.81      0.65      0.43     17104
* avg / total       0.99      0.68      0.62      0.81      0.65      0.43     17205

## Model 3 - ClusterCentroids Undersampling
Resampling Produced -  246 for both "low_risk" and "high_risk" target categories. 
Balanced_Accuracy = 0.6537310478007576
Confusion Matrix: 
array([[   67,    34],
       [10217,  6887]]
Classification_report
                 pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.66      0.40      0.01      0.52      0.27       101
   low_risk       1.00      0.40      0.66      0.57      0.52      0.26     17104

avg / total       0.99      0.40      0.66      0.57      0.52      0.26     17205

## Model 4 - Combination of Over and Under Sampling
Resampling Produced -  46660 "low_risk" rows and 51359 "high_risk" rows. 
Balanced_Accuracy = 0.5330103432466726
Confusion Matrix: 
array([[   67,    34],
       [ 7043, 10061]
Classification_report
                 pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.66      0.59      0.02      0.62      0.39       101
   low_risk       1.00      0.59      0.66      0.74      0.62      0.39     17104

avg / total       0.99      0.59      0.66      0.74      0.62      0.39     17205




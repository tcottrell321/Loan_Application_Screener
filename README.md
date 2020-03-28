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
|75  |26  |
|7216|9888|      

## Model 2 - SMOTE Oversampling 
* Resampling Produced - 51366 for both "low_risk" and "high_risk" target categories. 
* Balanced_Accuracy = 0.6537310478007576
* Confusion Matrix: 
|63 |38|,
|5410 |11694|


## Model 3 - ClusterCentroids Undersampling
Resampling Produced -  246 for both "low_risk" and "high_risk" target categories. 
Balanced_Accuracy = 0.6537310478007576
Confusion Matrix: 
|67 |34|
|10217 |6887 |

## Model 4 - Combination of Over and Under Sampling
Resampling Produced -  46660 "low_risk" rows and 51359 "high_risk" rows. 
Balanced_Accuracy = 0.5330103432466726
Confusion Matrix: 
|67 |34|
| 7043 |10061|

## General Comments On Using ML To Screen Loans
From the LendingCLub standpoint, using an ML Model to screen out high-risk loan applications - who ultimately might default on the loan - is important to keep profits high. One defaulted loan can wipe out profits from tens or even hundreds of good loans depending upon the defaulted loan amount so its important that Precision be as high as posssible -- resulting in less loans to high-risk applicants. 

On the other hand, a model that is "too sensitive" can screen out good borrowers, reducing the amount of processed business and overall customers. In general, ML models tradeoff Precision in "catching" the negative case while rejecting the "good case" so its important to understand the business tradeoffs that come with choosing the model -- or have backup business review processes to do a second screening of ML results. ML can therefore be used to increase the efficiency of Human UnderWriters who will do the secondary review. 

## Analysis on Comparing Models
Given that our Target Output is not continuous data, using a Balanced Accuracy Score Comparison can produce very erroneous conclusions. Never-the-less, looking at the accuracy, the model with the highest accuracy was the Random Oversampling model being correct 66% of the time. The SMOTE and ClusterCentroids models both came in at 65.3%, with the Combination Model producing the lowest accuracy the models varied slightly in their accuracy from a low of of 53.3%. 

* Note: A better measure is the Confusion Matrix and Classification Report. Please refer to the printouts in the Notebook showing specific output for each model. 

Overall the Random Sampling Model appears to be the best model, letting the least number of high_risk candidates be classified as False Negatives (26 loan apps) and also had a lower (not lowest) number of low_risk applicants misclassified as high risk (7216). 

The SMOTE Oversampling Model was much better at not rejecting low-risk applicatants (only 5410) but did let more high-risk borrowers thorugh (38). 

The SMOTEENN Combination Model had the next best - letting slightly more high-risk candidates through (34) but like Random Sampling, rejected 7043 good borrowers.  

The worst performing model appears to be the ClusterCentroids with high approval of bad borrowers (34) and also high rejection of good borrorwers (10217). 

## Recommendations
If the objective of the LendingClub is to maximize marketshare then choosing the SMOTE Oversampling will provide the highest number of approvals for low-risk loan applicants with a slight hit on profitability per loan by accepting a higher rate of approvals for high-risk applicants. 

If they are trying to maximize profitability instead of markeshare, then choosing the Random Sampling Model might be best to minimize the number of approvals on high-risk candidates at some sacrifice of rejecting good candidates. 

Regardless of the model, having Human Underwriters to check all loan applications either 100% or by statistical sampling -- should be put in place as a backup review process. It is also important from a regulation standpoint, that all candidates are given fair treatment under the law and that the ML Model does not contain unlawful bias's by rejecting specific "classes" of applicants. 





Given that our Target Output is not continuous data, using a Balanced Accuracy Score Comparison can produce very erroneous conclusions. Never-the-less, looking at the accuracy, the model with the highest accuracy was the Random Oversampling model being correct 66% of the time. The SMOTE and ClusterCentroids models both came in at 65.3%, with the Combination Model producing the lowest accuracy the models varied slightly in their accuracy from a low of of 53.3%. 

A better measure is the Confusion Matrix. 


# Loan_Application_Screener

## Background
LendingClub is a peer-to-peer lending company in which private investors can invest their own money by lending it to other individuals. Investors put in a sum of money and then through the LendingClub platform, distribute their funds across multiple loans with different credit risks. Investors earn a portion of the interest charged to the borrowers - with the potential to earn much higher interest rates overall than paid for deposit funds at banks, credit unions, or money markets. However, the risk of borrower default can be much higher as well. To aid the Investor, each potential borrower is evaluated for their overall risk of "paying back the loan" by LendingClub. Lending Club makes millions of individual loans and keeps original loan application files and if the borrower ultimately paid back the loan or defaulted. 

This data is stripped of borrower ID information making it anonymous, but the variables during application are kept in tact so they can be evaluated by a machine learning model. To facilitate building and testing various ML Models, a CSV file was provided for "train" and "test" data. 

## Challenges
Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, we are employing different techniques to train and evaluate models with unbalanced classes, then comparing their precision, recall, and overall F1 scores to see if any of the models standout as superior to the others, making a recommendation on possible use.  

## Resources: 
* 3.6 Python plus Numpy library
* Imbalanced-learn and scikit-learn libraries 

## Results By Model 

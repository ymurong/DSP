## Planing 

### explorative analysis

Any missing values, outliers, bias?

Related Notebooks:
[descriptive](descriptive.ipynb)

### basic visualization & inferential analysis

This would allow us to select the most relevant features (Correlation Matrix, PCA)

Related Notebooks:
* [Time Independant](visualization_time_independant.ipynb)
  * Correlation Matrix (Cramer V, Theil U) between categorical features and Fraud
  * PCA Viz to detect patterns between categorical features and Fraud
  * Euro Amounts Rank Sum Test
  * (WIP) Euro Amounts / Fraud per account outlier existance? 

* [Time Dependant](visualization_time_dependant.ipynb) (construct new features based on given dataset)
  * cumulative frauds for a given window time range (same ip, same account)
  * eur amounts outlier for a given past time range





## UvA Deadlines

26-30 Sept: meeting with the supervisor to discuss the current set of ideas`

24-28 Oct: End ideation meeting: Groups arrange a meeting with the supervisor and preferably with stakeholders to discuss the idea they chose to work out until the end of the project

21-25 nov: •Mid-proto meeting. Groups arrange a meeting with the supervisor to discuss the current state of the prototype

1-9 Dec: End-proto meeting. Groups arrange a meeting with the supervisor (and preferably with stakeholders) to discuss the way how they intend to ***implement***
 and ***evaluate*** the current prototypec 
 <br />
 
## Overall Business Questions 
1) how could we measure quantitively the consequences of FP?  High FP → legal consumers did not succeed in payments → more complaints of adyen clients →  → clients moving to other services?  

2) Similarly, how could we measure quantitively the consequences of FN  (through sum of chargeback?)

3) How could we balance the economic consequences of FN and FP? as unfortunately, these two things are normally contradicted → **Less FP leads to  Higher FN and vice versa →** need to find a comfortable balance so that we get less FN as possible and keep the FP at an acceptable rate → the model needs to be adjusted so that we achieve this sweet spot. (**Precision-Recall Balance**)
<br />

## Model Measurements Proposal

### blocked transactions

Among all blocked transactions by Adyen system,  we could have TP & FP cases. 

TP: there is indeed a fraud and our model has successfully detected it 

FP: there is actually no fraud but our model blocked it

**Precision rate  = TP / (TP+FP) → should be high**

Obviously here we need to **lower the FP** as it would damage the reputation of adyen system →  the higher the precision rate the better. 
<br />

### non-blocked transactions
Among all non-blocked transactions by Adyen system,  we could have TN & FN cases. 

TN: there is no fraud and our system let it pass as it should be 

FN: there is actually fraud but our system ignored it 

#### Recall rate  = TP / (TP+FN)  → should be high

Obviously here we need to **lower the FN** as it would lead to chargebacks →  the higher the recall rate the better. 
<br />

### metrics
**F1 score:** F1 = 2*PRECISON*RECALL / (PRECISION + RECALL)

it can only be high if recall and precision rate are both high

**PR curve**: adjust the decision boundary in order to achieve the Precision-Recall balance.




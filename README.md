# Table of Contents

- [Table of Contents](#table-of-contents)
  - [UvA Deadlines](#uva-deadlines)
  - [Presentation Slides](#presentation-slides)
  - [Explorative Analysis](#explorative-analysis)
  - [Visualization \& Inferential Analysis](#visualization--inferential-analysis)
    - [Feature Engineering](#feature-engineering)
    - [Classifier Training/Evaluation](#classifier-trainingevaluation)
    - [Backend](#backend)
    - [Dashboard](#dashboard)

## UvA Deadlines

26-30 Sept: meeting with the supervisor to discuss the current set of ideas`

24-28 Oct: End ideation meeting: Groups arrange a meeting with the supervisor and preferably with stakeholders to discuss the idea they chose to work out until the end of the project

21-25 nov: •Mid-proto meeting. Groups arrange a meeting with the supervisor to discuss the current state of the prototype

1-9 Dec: End-proto meeting. Groups arrange a meeting with the supervisor (and preferably with stakeholders) to discuss the way how they intend to ***implement***
 and ***evaluate*** the current prototype.

## Presentation Slides

[Ideation Phase Slides Link](https://docs.google.com/presentation/d/1JcW1Lxu41uvTsr47vixLqUnApinEqo_DoCd_Om2QcVE/edit#slide=id.p)


[Prototype Phase Slides Link](https://docs.google.com/presentation/d/1pRIDMuRsB5ZIwAhkmasCoDJcMUB6IIQHSfBdcfv4s7E/edit#slide=id.g192ae5ba20d_2_0)



## Explorative Analysis
Any missing values, outliers, bias?

Related Notebooks:
[descriptive](descriptive.ipynb)

## Visualization & Inferential Analysis
This would allow us to select the most relevant features and possibly construct new features that correlate with fraud based on given dataset

Related Notebooks:
* [Time Independant](visualization_time_independant.ipynb)
  * Correlation Matrix (Cramer V, Theil U) between categorical features and Fraud
  * Euro Amounts Rank Sum Test
  * PCA Viz to detect patterns between features and Fraud (eur_amount included and excluded)
  * (WIP) Risk Score based on historical transactions (new feature)
    * Would a client/ip whose transaction amount distribution differs from the general non fraud distribution indicates higher risk of fraud (Odds ratio) ?
      Justify this by searching for the account/ip that don't have the same distribution (by hypothesis testing) and visualize its fraud cases. 
      Based on this, possibly construct a risk score (high, midium, low) for each account/ip. Test risk score correlation with the fraud.
    * Would a client/ip who had fraud before indicates higher risk of fraud? weighted historical frauds counts

* [Time Dependant](visualization_time_dependant.ipynb) (construct new features based on given dataset)
  * cumulative frauds for a given window time range (same ip, same account)
  * eur amounts outlier for a given past time range


### Feature Engineering

Check the [FE_README.md](./feature-engineering/README.md) for details.

### Classifier Training/Evaluation

Check the [CLASSIFIER_README.md](./classifier/README.md) for details.


### Backend

Check the [BACKEND_README.md](./backend/README.md) for details. 

The backend provides online openapi documentation http://127.0.0.1:8000/docs 


### Dashboard

Check the [DASHBOARD_README.md](./dashboard-front/coreui/README.md) for details. 










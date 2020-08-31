# Bank-Customer-Churn-Prediction

## Prediciting Customer Churn for a European Retail Bank

From established enterprises to start-ups, good models for customer churn prediction are a vital necessity to:

- Better understand future revenue
- Understand areas where customer service needs improvement
- Target customer retention efforts and costs more effectively
- Find customers for whom existing marketing efforts have become ineffective

## The problem

- For all B2C businesses, including retail banks
  - Customer lifetime value is key
  - Loyalty is everything!

- Average churn for retail banks is 20% to 25% (Source:https://www.qualtrics.com/blog/customer-churn-banking/)
 
- Lost revenue from churned ex-customers is very hard to replace

## What can be done?

- Identify likely churners
- Launch targeted retention campaigns
  - Get frequent feedback
  - Improve customer service
  - Improve digital self-service 
  - Reduce fees & rates
  - Increase savings interest rates
  - Offer a wider variety of financial products

  All of the above cost a varying amount of $$$$. How much should we spend? 

  #### We need a Profit Curve! ####

## The Goals of this Project:

- Implement an optimal supervised machine learning model (or a pipeline of an ensemble of models) to best predict customer churn
- Derive a Profit Curve to understand the kind of budget that would be justifiable to execute a retention program

## The Process

- Data: Get enough labelled data
- EDA: Explore and analyze the data
- Score: Set a metric to evaluate the models
- Base model: Iterate through various model options to find the best base model
- Tune Model: Gridsearch for the best hyperparameters
- Profit Curve: Plug in cost-benefit numbers to find optimal retention program budgets

## The Data

- Data Source
  - Neuraldesigner (https://www.neuraldesigner.com/learning/examples/bank-churn#DataSet)
- Description of Data: 
  - Anonymized European Retail Bank customer data with ~10,000 rows and 12 features 
    - customer_id
    - credit_score
    - country
    - gender
    - age
    - tenure
    - balance
    - products_number
    - credit_card
    - active_member
    - estimated_salary
  - And a churn (or not) label
  
  - The entire data can be downloaded from https://www.neuraldesigner.com/files/datasets/bank_churn.csv
  - A small sample of the data is in the repo at 'data/bank_churn_shard.csv'

## EDA

- Overall churn percentage : 20.37%
- A balanced, well-crafted sample
- Bar charts, histograms and scatter matrix plots of various aspects of the data are to be fould in the jupyter notebook in the folder /Notebook
- No obvious correlations between features was observed
- Looking at churn vs. individual features, customer age, balance and number of products a customer uses had some clear differences in distributtion between churners and non-churners. All all other features no difference in distribution was observed 

## Model Scoring

Recall: is the metric we are most interested in because false negatives (not finding churners) is what we most want to minimize
Precision: is also interesting
Accuracy: should not be too low

## Base Model

- Started with 3 options:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
- First results:
  - Recall Scores were poor for all the models:
  - Then balanced the classes using:
    - Undersampling of the majority class
    - Random oversampling of the minority class (with replacement)
    - Synthetic Minority Class Oversampling Technique(SMOTE)
    - A combination of Undersampling and SMOTE
    
    Detail results for each of these techniques for all three models are to be found in the jupyter notebook in the /Notebook folder in this repo, and are summarized in the slide presentation in the /Presentation folder in this repo
- Following the above, the GradientBoosting Model with random minority class oversampling was chosen as the base model (based on consistently best Recall scores)
- A check on whether the model would improve after dropping least important features based on feature importances was done, but did not improve scores

## Tuning the base model to find the best model

- Several rounds of GridSearch with various parametrs (detailed in the jupyter notebook to be found in the folder /Notebook in this repo) were done but Recall score could not be improved)
  - Results for GradientBoostingClassifier
    - Gridsearched best model Recall: 0.709
    - Base model Recall: 0.748
- The base model was thus adopted as the best model to proceed with

## Profit Curve

Assumptions:

- A customer who leaves the bank will result in average annual revenue loss of $1,000 ... this was based on:
  - Retail Lending Interest Rate: 1.80% (Source: European Central Bank 2019)
  - Retail Deposit Interest Rate: 0.37% (Source: European Central Bank 2019)
  - Average balance/customer (in sample): $76,485
  - Average earning/customer (in sample): $76,486 * (0.18 - 0.037) =  $1,094
  - Cross-validated with Annual Report of JP Morgan Chase from 2011 which had = $912 (Source: http://investor.shareholder.com/jpmorganchase/annual.cfm)

- $200 per likely-to-churn-customer, which could pay for:
  - Fee reductions
  - Better savings interest rates
  - A birthday credit from the bank
  - Individual outreach by customer service or email
  - Glossy brochure or other token gifts
  - Etc.

- Retention Program is 100% effective

With the assumptions for the cost benefit matrix, and the calculated confusion matrix, the profit curve was plotted. This is to be found in the jupyter notebook in the folder /Notebooks in this repo
The findings were that, with a $200 per customer retention program
- The maximum incremental profit would be $68,400
- At a conservative churn probability threhold of 0.25

## Next Steps
- Try Neural Network models to see if Recall can be improved
- All customers are not the same, consider more retention program spending on High-Value vs Low-Value customers
- Use Uplift Models to skew the retention program spend towards customers that are predicted to be more persuadable
- Explore the effects on the Profit Curve for various levels of retention program costs

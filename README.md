# About the Project

## Project Goals

My goal is to identify key drivers for logerror by utilizing clustering.


## Project Description

A home is often the largest and most expensive purchase a person makes in his or her lifetime. Ensuring homeowners have a trusted way to monitor this asset is incredibly important. One considers several aspects while purchasing a home, the size, how many rooms are available, and many more.

Zillow is a popular estimator for house evaluation available online.  Zillow's Zestimate allows the homebuyers to search for a home that satisfies their location, area, budget, etc.

In this project we want to predict the property tax assessed values ('taxvaluedollarcnt') for single family properties. The focus will be the single unit properties that had a transaction during 2017.


### Initial Questions

- What is the relationship between bedroom count and taxvaluedollarcount?
    - Is it a linear relationship or is there no relationship?
    
- What is the relationship between bathroom count and taxvaluedollarcount?
    - Is it a linear relationship or is there no relationship?

- What is the relationship between square feet and taxvaluedollarcount?
    - Is it a linear relationship or is there no relationship?




### Data Dictionary

| Variable            |     Description  |     
| ----------------    | ------------------ |
|bedroom_cnt          | Number of bedrooms in each home |
|bathroom_cnt         | Number of bathrooms in each home. Can be half bathrooms and there will not be rows with 0 bathrooms |
|pool_cnt             | Number of pools in each home |
|nbr_stories          | Number of stories in each home |
|assessed_tax_value   | The target variable. Assessed property tax value of each home |
|year_built           | The year the home was built  |
|fips                 | Coding used to identify the county in which the home is located in |
|comb_sq_ft           | An amalgamation of three different columns from the Zillow database. This column is a sum of columns: basementsqft, garagetotalsqft, and calculatedfinishedsquarefeet |
|location             | A mapped fips column reflecting the name of the county of the fips code represents |




## Steps to Reproduce

- Create an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow table. Store that env file locally in the repository.
- Clone my repo (including an acquire.py and prepare.py) (confirm .gitignore is hiding your env.py file)
- Libraries used are pandas, matplotlib, seaborn, numpy, sklearn.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.

### Plan

Plan - Acquire - Prepare - Explore - Model - Deliver

- Wrangle
    - Acquire data by using a SQL query to Zillow table in the mySQL database.
    - Prepare data by doing a cleanup of null values, duplicates, removed unnecessary outliers.
    - We will create a function that we can reference later to acquire and prepare the data by storing the function in a file name wrangle.py
    - We will split our data to a train, validate, and test
- Explore
    - Create visualizations of data to pin point key drivers related to logerror
    - Create a visualizations correlating to hypotheses statements
    - Run at least two statistical tests that will support whether the hypothesis has been rejected or not
- Modeling
    - Establish baseline
    - Ensure models are tested on appropriate validate and test datasets
    - Determine best performing model and test on test dataset# clustering-project
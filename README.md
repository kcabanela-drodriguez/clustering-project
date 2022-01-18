# About the Project

## Project Goals

My goal is to identify key drivers for logerror by utilizing clustering.


## Project Description

A home is often the largest and most expensive purchase a person makes in his or her lifetime. Ensuring homeowners have a trusted way to monitor this asset is incredibly important. One considers several aspects while purchasing a home, the size, how many rooms are available, and many more.

Zillow is a popular estimator for house evaluation available online.  Zillow's Zestimate allows the homebuyers to search for a home that satisfies their location, area, budget, etc.

In this project we want to uncover what the drivers of the error are in Zillow's Zestimate. The focus will be the single unit properties that had a transaction during 2017.


### Initial Questions

- Is there a relationship between logerror and longitude and latitude?

- What is the relationship between What is the relationship between bedroom count and logerror?
    
- What is the relationship between square feet and logerror?

- How different are the logerrors for the three counties in the dataset?




### Data Dictionary

| Variable            |     Description  |     
| ----------------    | ------------------ |
|bathroomcnt          | Number of bathrooms in each home.  Can be half bathrooms and there will not be rows with 0 bathrooms |
|bedroomcnt         | Number of bedrooms in each home |
|calculatedfinishedsquarefeet             | Square feet of property |
|latitude          | Latitude of home location |
|longitude   | Longitude of home location |
|lotsizesquarefeet | Square feet total of home lot |
|rawcensustractandblock           | Raw census tract and block id combined  |
|regionidzip                 | Home zip code |
|roomcnt           | Total number of rooms in the home |
|yearbuilt             | The year the home was built |
|structuretaxvaluedollarcnt          | The assessed value of the home |
|taxvaluedollarcnt          | Tax value assessed |
|landtaxvaluedollarcnt          | Tax value of land where home is located |
|taxamount          | Tax amount paid on the property |
|censustractandblock          | Census tract and block id combined |
|logerror          | Target variable |
|county          | County where home is located |




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
- Delivery
    - Final Report in Jupyter Notebook
    - README with project details
    - Python modules with acquire and prepare functions

### Import Section ### 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from env import user, password, host
from os.path import exists

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

#function to acquire zillow data from codeup database --> place in wrangle.py module
def acquire_zillow():
    """
    Purpose
        To do initial acquisition of zillow data

    Parameters
        None

    Returns
        data frame containing the zillow real estate data
    """
    #create url to access DB
    url = f"mysql+pymysql://{user}:{password}@{host}/zillow"

    #sql to acquire data. 
    sql = """
    SELECT bedroomcnt as bedrooms, 
        bathroomcnt as bathrooms, 
        calculatedfinishedsquarefeet as area, 
        taxvaluedollarcnt as home_value,
        yearbuilt as year_built, 
        taxamount as tax_amount, 
        fips as county
    FROM properties_2017
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    LEFT JOIN predictions_2017 USING (parcelid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential") 
        AND transactiondate like '%%2017%%'
    """

    #read the data in a dataFrame. will use pre-existing csv file if available;
    if exists('zillow_data.csv'):
        df = pd.read_csv('zillow_data.csv').drop(columns='Unnamed: 0')
    else:
        df = pd.read_sql(sql, url)

    #drops null values from df    
    df = df.dropna(axis=0)

    return df

#engineer features
def engineer_features(df):
    """
    Purpose
        To engineer features from data set for exploration and modeling
    Parameters
        df: dataframe containing zillow real estate data
    Returns
        df: dataframe containing proper features, and with unwanted columns removed
    """
    
    #### Decades: 
    #create list to hold labels for decades
    decade_labels = [x + 's' for x in np.arange(1870, 2030, 10)[:-1].astype('str')]

    #assign decades created from range to new decades column in dataset and apply labels
    df['decades'] = pd.cut(df.year_built, np.arange(1870, 2030, 10), labels=decade_labels, ordered=True)
    #age 
    df['home_age'] = (2022 - df.year_built).astype('int')

    #### Home Size
    #use quantiles to calculate subgroups and assign to new column
    q1, q3 = df.area.quantile([.25, .75])
    df['home_size'] = pd.cut(df.area, [0,q1,q3, df.area.max()], labels=['small', 'medium', 'large'], right=True)

    #### Estimated Tax Rate
    df['est_tax_rate'] = df.tax_amount / df.home_value

    #remove columns
    df = df.drop(columns=['year_built', 'tax_amount'])

    return df

# function for removing outliers
def remove_outliers(df, k, col_list):
    """ 
    Purpose
        remove outliers from a list of columns in a dataframe and return that dataframe
    Parameters
        df: dataframe
        k: int, to help control tolerance level for outlier
        col_list: list, list of columns in which to find outliers
    """

    # total number of observations
    num_obs = df.shape[0]
        
    # Create a column that will label our rows as containing an outlier. sets default value
    df['outlier'] = False

    # loop through the columns provided to find appropriate values and labels
    for col in col_list:

        # find quartiles
        q1, q3 = df[col].quantile([.25, .75])  
        
       # get interquartile range
        iqr = q3 - q1

       # find upper/lower bounds 
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label as needed. 
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    # set dataframe to dataframe w/o the outliers
    df = df[df.outlier == False]

    # drop the outlier column from the dataFrame. no longer needed
    df.drop(columns=['outlier'], inplace=True)

    # print out number of removed observations
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df


#### Prepare ####
def prep(df):
    """
    Purpose
        To return dataset for exploration

    Parameters
        df: dataframe to perform desired operations on

    Returns
        adjusted dataframe
        train, validate, and test subsets
        scaled versions of train, validate, and test subsets
    """

    #change county fips codes to county names
    df.county = df.county.map({6037: 'LA County', 6059: 'Orange County', 6111: 'Ventura County'})

    #engineer features
    df = engineer_features(df)

    #remove outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'home_value', 'home_age', 'est_tax_rate'])

    #change datatypes of categorical columns
    df.county = df.county.astype('object')

    #train_test_split
    train_validate, test = train_test_split(df, test_size=.2, random_state=514)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=514)
    
    #create scaler object
    scaler = MinMaxScaler()

    # create copies to hold scaled data
    train_scaled = train.copy(deep=True)
    validate_scaled = validate.copy(deep=True)
    test_scaled =  test.copy(deep=True)

    #create list of numeric columns for scaling
    num_cols = train.select_dtypes(include='number')

    #fit to data
    scaler.fit(num_cols)

    # apply
    train_scaled[num_cols.columns] = scaler.transform(train[num_cols.columns])
    validate_scaled[num_cols.columns] =  scaler.transform(validate[num_cols.columns])
    test_scaled[num_cols.columns] =  scaler.transform(test[num_cols.columns]
    )

    #plot basic visuals of column counts, boxplots for numeric columns, and 
    plt.figure(figsize=(20,5))
    i=1

    for col in df.columns:
        plt.subplot(1, len(df.columns), i)
        df[col].hist(bins=5)
        plt.title(col)
        i += 1
    plt.tight_layout()
    plt.show()

    #boxplot visual
    plt.figure(figsize=(20,5))
    i=1

    for col in df.select_dtypes(include='number').columns:
    
        plt.subplot(1, len(df.columns), i)
        sns.boxplot(data=df[col])
        plt.title(col)
        i += 1
    plt.tight_layout()
    plt.show()

        
    return df, train, train_scaled, validate, validate_scaled, test, test_scaled
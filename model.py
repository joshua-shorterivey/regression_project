#### Import Section
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from wrangle import prep, acquire_zillow

from itertools import product
from scipy.stats import levene , pearsonr, spearmanr, mannwhitneyu, f_oneway, ttest_ind
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, f_regression, SelectKBest

def modeling_prep (train, train_scaled, validate, validate_scaled, test, test_scaled):
    # create X,y for train, validate and test subsets
    X_train = train_scaled.drop(columns='home_value')
    y_train = train.home_value
    X_val = validate_scaled.drop(columns='home_value')
    y_val = validate.home_value
    X_test = test_scaled.drop(columns='home_value')
    y_test = test_scaled.home_value

    #shift y subsets into a data frame
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)

    #add baseline predictions
    y_train['pred_median'] = y_train.home_value.median()
    y_val['pred_median'] = y_val.home_value.median()
    y_test['pred_median'] = y_test.home_value.median()


    #get dummies for X subsets
    X_train = pd.get_dummies(X_train, columns=['county', 'home_size', 'decades'], drop_first=True)
    X_val = pd.get_dummies(X_val, columns=['county', 'home_size', 'decades'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['county', 'home_size', 'decades'], drop_first=True)

    return X_train, y_train, X_val, y_val, X_test, y_test

def select_kbest(X, y, k): 
    # initilize selector object
    f_selector = SelectKBest(f_regression, k=k)

    #fit object --> will find top 2 as requested
    f_selector.fit(X, y)

    # create mask
    feature_mask = f_selector.get_support()

    # use mask to show list of feature support
    f_top_features = X.iloc[:,feature_mask].columns.tolist()

    return f_top_features


def rfe (X, y, n):

    #initialize  regression object
    lm = LinearRegression()

    # initilize RFE object with n features
    rfe = RFE(lm, n_features_to_select=n)

    #fit object onto data
    rfe.fit(X, y)

    #create boolean mask for columns model selects 
    feature_mask = rfe.support_

    # use mask to show list of selected features
    rfe_top_features = X.iloc[:, feature_mask].columns.tolist()

    return rfe_top_features

def get_features(X_train, y_train):
    all_features = list(X_train.columns)

    req_features  = ['area', 'bedrooms', 'bathrooms']

    feat_set1  = ['est_tax_rate', 'area']

    feat_set2 = ['est_tax_rate', 'area', 'home_age']

    feat_set3 = ['est_tax_rate', 'area', 'home_age', 'county_Orange County', 'county_Ventura County']

    feat_set4 = ['est_tax_rate', 'area', 'bathrooms', 'home_age', 'county_Orange County', 'county_Ventura County']

    feat_rfe = rfe(X_train, y_train.home_value, 4)

    feat_sk_best = select_kbest(X_train, y_train.home_value, 4)

    feat_combos = [all_features, req_features, feat_set1, feat_set2, feat_set3, feat_set4, feat_sk_best, feat_rfe]

    return feat_combos

def pf_mod(X, y, selectors, fit_train=None, fit_y_train=None):


    pf_descriptions = pd.DataFrame({}, columns=['Name','RMSE', 'Features', 'Parameters'])

    for idx, combo in enumerate(selectors):
        pf = PolynomialFeatures(degree=combo[1])

        lm = LinearRegression(normalize=True)

        if fit_train is not None:
            fit_pf = pf.fit_transform(fit_train[combo[0]])
            X_pf = pf.transform(X[combo[0]])  
            lm.fit(fit_pf, fit_y_train.home_value)
        else:
            X_pf = pf.fit_transform(X[combo[0]])
            lm.fit(X_pf, y.home_value)

        model_label = f'Polynomial_{idx+1}'

        #predict train
        y[model_label] = lm.predict(X_pf) 

        #calculate train rmse
        rmse = mean_squared_error(y.home_value, y[model_label], squared=False)

        # print(f'{model_label} with degree: {combo[1]} \n\
        #     Features: {combo[0]} \n\
        #     RMSE: {rmse}\n')
        
        description = pd.DataFrame([[model_label, rmse, combo[0], f'Degree: {combo[1]}']], columns=['Name', 'RMSE', 'Features', 'Parameters'])
        pf_descriptions = pd.concat([pf_descriptions, description])

    return pf_descriptions
def ols_mod(X, y, selectors, fit_x_train=None, fit_y_train=None):

    ols_descriptions = pd.DataFrame({}, columns=['Name','RMSE', 'Features', 'Parameters'])

    for idx, features in enumerate(selectors):  
        lm = LinearRegression()
    
        model_label = f'OLS_{idx+1}'

        if fit_x_train is not None:
            lm.fit(fit_x_train[features], fit_y_train.home_value)
        else:   
            lm.fit(X[features], y.home_value)

        #predict train
        y[model_label] = lm.predict(X[features]) 

        #calc trian rmse
        rmse = mean_squared_error(y.home_value, y[model_label], squared=False)

        # print(f'{model_label} with LinearRegression\n\
        #     Features: {features}\n\
        #     RMSE: {rmse_train}\n')

        description = pd.DataFrame([[model_label, rmse, features, 'N/A']], columns=['Name', 'RMSE', 'Features', 'Parameters'])
        ols_descriptions = pd.concat([ols_descriptions, description])

    return ols_descriptions
def lars_mod(X, y, selectors):

    lars_descriptions = pd.DataFrame({}, columns=['Name','RMSE', 'Features', 'Parameters'])

    for idx, selector in enumerate(selectors):  
        lars = LassoLars(alpha=selector)
    
        model_label = f'LARS_{idx+1}'

        #fit mode 
        lars.fit(X, y.home_value)

        #predict train
        y[model_label] = lars.predict(X) 

        #calc trian rmse
        rmse = mean_squared_error(y.home_value, y[model_label], squared=False)

        description = pd.DataFrame([[model_label, rmse, 'all', f'Alpha: {selector}']], columns=['Name', 'RMSE', 'Features', 'Parameters'])
        lars_descriptions = pd.concat([lars_descriptions, description])

    return lars_descriptions

def GLM_mod(X, y, selectors):
    
    glm_descriptions = pd.DataFrame({}, columns=['Name','RMSE', 'Features', 'Parameters'])

    for idx, combo in enumerate(selectors):  
        glm = TweedieRegressor(power=combo[0], alpha=combo[1])
    
        model_label = f'GLM_{idx+1}'

        #fit mode 
        glm.fit(X, y.home_value)

        #predict train
        y[model_label] = glm.predict(X) 

        #calc trian rmse
        rmse = mean_squared_error(y.home_value, y[model_label], squared=False)

        # print(f'{model_label} with Tweedie \n\
        #     Power: {combo[0][0]}, alpha: {combo[0][1]}\n\
        #     Features: {combo[1]} \n\
        #     RMSE: {rmse_train}')
        description = pd.DataFrame([[model_label, rmse, '-', f'Power,Alpha: {combo}']], columns=['Name', 'RMSE', 'Features', 'Parameters'])
        glm_descriptions = pd.concat([glm_descriptions, description])

    return glm_descriptions

def train_score(X_train, y_train): 

    feat_combos = get_features(X_train, y_train)

    #create a lists of parameters
    pf_parameters = [2,3]
    lars_parameters = [.25, .5, .75, 1]
    glm_parameters = [(0,1), (0,.25), (0,.5), (0,.75), (0,1), (1,1), (1,.25), (1,.5), (1,.75), (1,1)]

    #use list with product to create tuples of feature/parameter combination to feed into model
    pf_selectors = list(product(feat_combos, pf_parameters))

    #run ols model with feature combinations
    pf_descriptions = pf_mod(X_train, y_train, pf_selectors)
    olf_descriptions = ols_mod(X_train, y_train, feat_combos)
    lars_descriptions = lars_mod(X_train, y_train, lars_parameters)
    glm_descriptions = GLM_mod(X_train, y_train, glm_parameters)

    rmse_train = mean_squared_error(y_train.home_value, y_train.pred_median, squared=False)

    model_descriptions = pd.DataFrame([['pred_median', rmse_train, 0, 'N/A', 'N/A']], columns=['Name','RMSE', 'r^2 score','Features', 'Parameters'])

    #create df for model scores on the train scores
    model_scores = pd.DataFrame({}, columns=['Model', 'r^2 score'])
    model_scores = model_scores.set_index('Model')
    model_descriptions = pd.concat([model_descriptions, pf_descriptions, lars_descriptions, olf_descriptions, glm_descriptions])
    model_descriptions = model_descriptions.set_index('Name')
    for idx, model in enumerate(y_train.drop(columns='home_value').columns):
        model_descriptions.loc[model, 'r^2 score'] = explained_variance_score(y_train['home_value'], y_train[model])



    return round(model_descriptions,2)







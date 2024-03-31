"""Classes of helper functions for data prep in a Sklearn pipeline for a Logistic regression model"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split


class PipelineHelper:
        
    # generate full dataframe from csv, numeric and categorical column names
    def generate_df_cols(csv_path, target):
        
        # read the csv into a dataframe
        df = pd.read_csv(csv_path, delimiter = ';')

        # separate numeric, categorical and target columns
        num_columns = df.select_dtypes(['integer', 'float']).columns
        cat_columns = df.select_dtypes(['object']).drop(columns = target).columns
        
        # print column types
        print("Numeric columns are {}.".format(", ".join(num_columns)))
        print("Categorical columns are {}.".format(", ".join(cat_columns)))
        
        return df, cat_columns, num_columns

    # split the data with a 90/10 training vs test
    def split_data(df, target):
        # split the data into training and testing data as 90-10 split
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = target), df[target], 
                                                        test_size = 0.10, random_state = 42)
        return X_train, X_test, y_train, y_test

    # after training pipeline get the precision and recall score
    def get_metrics(y_test, y_pred):
       
        precision_test = precision_score(y_test, y_pred, pos_label = 'yes') * 100
        recall_test = recall_score(y_test, y_pred, pos_label = 'yes') * 100
        print("Precision = {:.0f}% and recall = {:.0f}% on the validation data.".format(precision_test, recall_test))
        
        return precision_test, recall_test


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    #Defining the map function
    #def binary_map(feature):
     #   return feature.map({'Yes':1, 'No':0})

    # Encode binary categorical features
    #binary_list =  [ 'has_children', 'home_owner', 'college_degree', 'good_credit']
    #df[binary_list] = df[binary_list].apply(binary_map)

    
    #Drop values based on operational options
    if (option == "Online"):
        columns = ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 'has_children',
       'length_of_residence', 'marital_status', 'home_owner', 'college_degree', 'good_credit']
        #Encoding the other categorical categoric features with more than two categories
        #df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    elif (option == "Batch"):
        pass
        df = df[ ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 'has_children',
       'length_of_residence', 'marital_status', 'home_owner', 'college_degree', 'good_credit']]
        columns =  ['curr_ann_amt', 'days_tenure', 'age_in_years', 'income', 'has_children',
       'length_of_residence', 'marital_status', 'home_owner', 'college_degree', 'good_credit']
        #Encoding the other categorical categoric features with more than two categories
        #df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    else:
        print("Incorrect operational options")

    

    from sklearn.preprocessing import StandardScaler
    # Scale the features
    #scaler = StandardScaler()
    #df = scaler.fit_transform(df)
    return df   
    





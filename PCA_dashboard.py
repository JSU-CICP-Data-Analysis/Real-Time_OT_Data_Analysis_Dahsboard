import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import os
import time
pd.set_option('display.max_columns', None)


## Panel
import holoviews as hv
import hvplot.pandas
import panel as pn
from holoviews import opts 

## Param
import param
import panel as pn


## hvplot and streamz /Might not use
# import hvplot.streamz
# from streamz.dataframe import PeriodicDataFrame
import param


import threading
import time
# Kill previous thread
stop_thread=True

hv.extension('bokeh')
pn.extension('katex', sizing_mode="stretch_width")

# This function plots the wcss (within cluster sum of squares error) for all k-values 
    # and graphs it.  This allows us to choose a the best value to use for k-means
    # using the elbow method.  We then define a variable (N) to be that value.
def kmeansplot(df3_std):
    wcss = [] # this is a list where we will store the sum of squares 
    for i in range(1,df3_std.shape[1]+1):
      kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state= 42)
      kmeans_pca.fit(df3_std)
      wcss.append(kmeans_pca.inertia_) # adding the sum of squares to our list
    plt.figure(figsize=(10,8))
    plt.plot(range(1,df3_std.shape[1]+1),wcss, marker = 'o', linestyle = '--')
    plt.title('K-Means with PCA Clustering')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within Cluster Sum of Squares')
    plt.show()

# Once we've chosen N, this re-runs K-means using that value.
    # it creates 3 global variables that can be referred to later.
    # Normally, variables within a function can't be used outside of the function, but global variables can.
def kmeansapply(df3_std,df4_std, N):
    global kmeans3, kmeans4, centers
    kmeans = KMeans(n_clusters = N, init = 'k-means++', random_state = 42)
    kmeans.fit(df3_std)
    # We're going to "transform" or "predict" the data from our 2 datsets using the kmeans model we just built
    # What we're actually doing is finding the clusters based on the centroids we found from dataset df3_std
    kmeans3 = kmeans.predict(df3_std)
    kmeans4 = kmeans.predict(df4_std)
        # each of these gives us two arrays with only 1 dimension each (aka, a list)
        # The values tell us which cluster each row belongs to
    # save centers to dataframe
    centers = pd.DataFrame(kmeans.cluster_centers_)
    print("kmeans3:")
    print(kmeans3.shape)
    print(kmeans3)
    print("kmeans4:")
    print(kmeans4.shape)
    print(kmeans4)
    
# run before loop
def find_unary(df):
    unary_features = []
    for feature in df.columns:
        unique_values = df[feature].unique()
        if len(unique_values) == 1:
            unary_features.append(feature)
    unary_features.append("P_J280") # Add in a line to drop this problematic near-constant feature
    return unary_features
    
# Drop Unary Features
	# & Drop the problem P_J280
	# Also have function to test these (code in kmeans.ipynb I think)
def drop_unary(df):
    unary_features = []
    for feature in df.columns:
        unique_values = df[feature].unique()
        if len(unique_values) == 1:
            unary_features.append(feature)
    unary_features.append("P_J280") # Add in a line to drop this problematic feature
    df = df.drop(unary_features, axis=1)
    return df

# Better Scaling
# Find Binary features
def find_binary(df):
    binary_features = []
    for feature in df.columns:
        unique_values = df[feature].unique()
        if len(unique_values) == 2:
            binary_features.append(feature)
    non_binary_features = [i for i in df.columns if i not in binary_features]
    return binary_features, non_binary_features

# Define better scaling
def better_scaling(df):
	binary, non_binary = find_binary(df)
	# Separate binary and non-binary columns
	df_binary = df[binary]
	df_non_binary = df[non_binary]
	# Only scale non-binary columns
	scaler = StandardScaler()
	df_non_binary_scaled = pd.DataFrame(scaler.fit_transform(df_non_binary), columns=non_binary, index=df_non_binary.index)
	# Concatenate back with binary columns
	df_scaled = pd.concat([df_binary, df_non_binary_scaled], axis=1)
	return df_scaled

# Correct version: indexes attack end at i-1
def find_attacks(df, flag_column, datetime_column):
    # make output dataframe
    df_attacks = pd.DataFrame(columns=['indexes', 'DATETIME', 'attack_number'])
    # # save flag_column name using quotes
    # # Note: this won't work since you can't pass an undefined variable to a function
    # quoted_flag_column = "'{}'".format(flag_column)
    # initialize variables to keep track of wheteher it's the start or stop of the function
    attack_flag = False
    # initialize attack number variable
    attack_number = 0
    
    for i in range(len(df)):
        # Check if the Bin_Flag changes from 0 to 1 or from 1 to 0
        if df.iloc[i][flag_column] == 1 and not attack_flag:
            attack_number += 1
            df_attacks = pd.concat([df_attacks, 
                                    pd.DataFrame({'indexes': [i], 'DATETIME': [df.iloc[i][datetime_column]], 
                                                  'attack_number': [f"{attack_number}-start"]})], ignore_index=True)
            attack_flag = True
        elif df.iloc[i][flag_column] == 0 and attack_flag:
            df_attacks = pd.concat([df_attacks, 
                                    pd.DataFrame({'indexes': [i-1], 'DATETIME': [df.iloc[i-1][datetime_column]],
                                                  'attack_number': [f"{attack_number}-stop"]})], ignore_index=True)
            attack_flag = False

    return df_attacks

def find_and_fix_binary(df):
    binary_features = []
    binary_features_dict = {}
    for feature in df.columns:
        unique_values = df[feature].unique()
        if len(unique_values) == 2:
            binary_features.append(feature)
            zero_val = 0 if 0 in unique_values else min(unique_values)
            one_val = 1 if 1 in unique_values else max(unique_values)
            # Save them in dictionary to apply to attack dataframe and replace the original values in this dataframe

            binary_features_dict[feature] = (zero_val, one_val)
            df[feature] = df[feature].replace({zero_val: 0, one_val: 1})
            # If a binary feature had values other than 0,1, let us know, and what it was changed to
            if set(unique_values) != {0, 1}:
                print(f"Feature: {feature}")
                print(f"Value {zero_val} changed to 0")
                print(f"Value {one_val} changed to 1")
    non_binary_features = [i for i in df.columns if i not in binary_features]
    return binary_features, non_binary_features, binary_features_dict

# binary feature check for df4
def check_and_fix_binary(df, binary_feature_dict):
    for feature, (zero_val, one_val) in binary_feature_dict.items():
        unique_values = df[feature].unique()
        # Check if the feature is still binary
        if set(unique_values) <= {zero_val, one_val}:
            # Apply the same transformation as the first DataFrame
            df[feature] = df[feature].replace({zero_val: 0, one_val: 1})
        else:
        	raise ValueError(f"Feature '{feature}' is not binary in the second DataFrame.")
            
            
def get_data(file):
    """
    Get data from file..
    """
    df4 = pd.read_csv(file)
    df4['DATETIME'] = pd.to_datetime(df4['DATETIME'], format='%d/%m/%y %H')
    df4 = df4.set_index('DATETIME')
    df4 = df4.drop(columns=['Bin_Flag','ATT_FLAG','True_Flag'], axis=1)
    df4_full = df4.copy()

    # Run on df3 initially (before loop)
    df4_unary = find_unary(df4)

    # df4 = drop_unary(df4)
    df4 = df4.drop(columns=df3_unary, axis=1)

    binary, non_binary = find_binary(df4)
    column_order = binary + non_binary # for something later in in the NN code

    #Normalize non_binary data
    df4_std = better_scaling(df4)
    
    df4_std = df4_std[df3_std.columns]
    
    # Only works around 1750 rows and 35 columns if I keep features F_PU11 & S_PU11 F_PU6 and S_PU6
    scores_pca4 = pca.transform(df4_std)
    PCA4 = pd.DataFrame(scores_pca4, index=df4.index)
    
    return PCA4


# Make a pandas dataframe from Batadal dataset03 (Benign/Normal Operation Data)
df3 = pd.read_csv('Data/BATADAL_dataset03_flagged.csv')
df3_flagged = df3.copy()
df3_full = df3.copy()

df3 = df3.drop("DATETIME",axis=1)
df3 = df3.drop("ATT_FLAG",axis=1)
df3 = df3.drop("True_Flag",axis=1)#

# Run on df3 initially (before loop)
df3_unary = find_unary(df3)
df3 = drop_unary(df3)

binary, non_binary = find_binary(df3)
column_order = binary + non_binary # for something later in in the NN code

#Normalize non_binary data
df3_std = better_scaling(df3)


#pca for df3
pca = PCA(n_components=df3.shape[1])
pca.fit(df3_std)
scores_pca3 = pca.transform(df3_std)
PCA3 = pd.DataFrame(scores_pca3, index=df3.index)

# generating list of PCA components. 
columns = df3_std.columns.copy()
components = []
i = 0 

while i < len(columns):
    components.append(str(i))
    i += 1   

int_components = [int(x) for x in components]

# creating an Input class 
class InputFile(param.Parameterized):   # param.Parameterized is the Base class for named objects that support Parameters and message formatting.
    file_selector = param.FileSelector(path='/Users/clarence/CICP/Data/*')
    
    # panel method that will show selected value for param
    @param.depends("file_selector", watch=True)
    def panel(self):
        # print(type(self.file_selector))
        return pn.Column(self.param.file_selector)
    
    
class DataProcessing(InputFile): # The Multiplying class inherits the attributes of class Input
    
    # Will give certain options for the data at this time to show how you would want to visualize your data...
    # file_selector = param.FileSelector()
    # file_selector = param.String()
    # file_selector = param.String(default='/Users/cconner/Desktop/CC_BATADAL/Data/sim4.csv')
    std_threshold = param.Integer(default = 3, bounds=(2.00, 5.00), precedence=-1)
    list_select = param.Selector(default='Min/Max', objects=['Min/Max', 'Threshold'])
    
    def kmeansplot(df3_std):
        wcss = [] # this is a list where we will store the sum of squares 
        for i in range(1,df3_std.shape[1]+1):
          kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state= 42)
          kmeans_pca.fit(df3_std)
          wcss.append(kmeans_pca.inertia_) # adding the sum of squares to our list
        plt.figure(figsize=(10,8))
        plt.plot(range(1,df3_std.shape[1]+1),wcss, marker = 'o', linestyle = '--')
        plt.title('K-Means with PCA Clustering')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Within Cluster Sum of Squares')
        plt.show()

    # Once we've chosen N, this re-runs K-means using that value.
        # it creates 3 global variables that can be referred to later.
        # Normally, variables within a function can't be used outside of the function, but global variables can.
    def kmeansapply(df3_std,df4_std, N):
        global kmeans3, kmeans4, centers
        kmeans = KMeans(n_clusters = N, init = 'k-means++', random_state = 42)
        kmeans.fit(df3_std)
        # We're going to "transform" or "predict" the data from our 2 datsets using the kmeans model we just built
        # What we're actually doing is finding the clusters based on the centroids we found from dataset df3_std
        kmeans3 = kmeans.predict(df3_std)
        kmeans4 = kmeans.predict(df4_std)
            # each of these gives us two arrays with only 1 dimension each (aka, a list)
            # The values tell us which cluster each row belongs to
        # save centers to dataframe
        centers = pd.DataFrame(kmeans.cluster_centers_)
        print("kmeans3:")
        print(kmeans3.shape)
        print(kmeans3)
        print("kmeans4:")
        print(kmeans4.shape)
        print(kmeans4)

    # run before loop
    def find_unary(df):
        unary_features = []
        for feature in df.columns:
            unique_values = df[feature].unique()
            if len(unique_values) == 1:
                unary_features.append(feature)
        unary_features.append("P_J280") # Add in a line to drop this problematic near-constant feature
        return unary_features

    # Drop Unary Features
        # & Drop the problem P_J280
        # Also have function to test these (code in kmeans.ipynb I think)
    def drop_unary(df):
        unary_features = []
        for feature in df.columns:
            unique_values = df[feature].unique()
            if len(unique_values) == 1:
                unary_features.append(feature)
        unary_features.append("P_J280") # Add in a line to drop this problematic feature
        df = df.drop(unary_features, axis=1)
        return df

    # Better Scaling
    # Find Binary features
    def find_binary(df):
        binary_features = []
        for feature in df.columns:
            unique_values = df[feature].unique()
            if len(unique_values) == 2:
                binary_features.append(feature)
        non_binary_features = [i for i in df.columns if i not in binary_features]
        return binary_features, non_binary_features

    # Define better scaling
    def better_scaling(df):
        binary, non_binary = find_binary(df)
        # Separate binary and non-binary columns
        df_binary = df[binary]
        df_non_binary = df[non_binary]
        # Only scale non-binary columns
        scaler = StandardScaler()
        df_non_binary_scaled = pd.DataFrame(scaler.fit_transform(df_non_binary), columns=non_binary, index=df_non_binary.index)
        # Concatenate back with binary columns
        df_scaled = pd.concat([df_binary, df_non_binary_scaled], axis=1)
        return df_scaled

    # Correct version: indexes attack end at i-1
    def find_attacks(df, flag_column, datetime_column):
        # make output dataframe
        df_attacks = pd.DataFrame(columns=['indexes', 'DATETIME', 'attack_number'])
        # # save flag_column name using quotes
        # # Note: this won't work since you can't pass an undefined variable to a function
        # quoted_flag_column = "'{}'".format(flag_column)
        # initialize variables to keep track of wheteher it's the start or stop of the function
        attack_flag = False
        # initialize attack number variable
        attack_number = 0

        for i in range(len(df)):
            # Check if the Bin_Flag changes from 0 to 1 or from 1 to 0
            if df.iloc[i][flag_column] == 1 and not attack_flag:
                attack_number += 1
                df_attacks = pd.concat([df_attacks, 
                                        pd.DataFrame({'indexes': [i], 'DATETIME': [df.iloc[i][datetime_column]], 
                                                      'attack_number': [f"{attack_number}-start"]})], ignore_index=True)
                attack_flag = True
            elif df.iloc[i][flag_column] == 0 and attack_flag:
                df_attacks = pd.concat([df_attacks, 
                                        pd.DataFrame({'indexes': [i-1], 'DATETIME': [df.iloc[i-1][datetime_column]],
                                                      'attack_number': [f"{attack_number}-stop"]})], ignore_index=True)
                attack_flag = False

        return df_attacks

    def find_and_fix_binary(df):
        binary_features = []
        binary_features_dict = {}
        for feature in df.columns:
            unique_values = df[feature].unique()
            if len(unique_values) == 2:
                binary_features.append(feature)
                zero_val = 0 if 0 in unique_values else min(unique_values)
                one_val = 1 if 1 in unique_values else max(unique_values)
                # Save them in dictionary to apply to attack dataframe and replace the original values in this dataframe

                binary_features_dict[feature] = (zero_val, one_val)
                df[feature] = df[feature].replace({zero_val: 0, one_val: 1})
                # If a binary feature had values other than 0,1, let us know, and what it was changed to
                if set(unique_values) != {0, 1}:
                    print(f"Feature: {feature}")
                    print(f"Value {zero_val} changed to 0")
                    print(f"Value {one_val} changed to 1")
        non_binary_features = [i for i in df.columns if i not in binary_features]
        return binary_features, non_binary_features, binary_features_dict

    # binary feature check for df4
    def check_and_fix_binary(df, binary_feature_dict):
        for feature, (zero_val, one_val) in binary_feature_dict.items():
            unique_values = df[feature].unique()
            # Check if the feature is still binary
            if set(unique_values) <= {zero_val, one_val}:
                # Apply the same transformation as the first DataFrame
                df[feature] = df[feature].replace({zero_val: 0, one_val: 1})
            else:
                raise ValueError(f"Feature '{feature}' is not binary in the second DataFrame.")

    #This function should be watched for updates. 
    def get_data(self):
        """
        Get data from file..
        """
        df4 = pd.read_csv(self.file_selector)
        df4['DATETIME'] = pd.to_datetime(df4['DATETIME'], format='%d/%m/%y %H')
        df4 = df4.set_index('DATETIME')
        df4 = df4.drop(columns=['Bin_Flag','ATT_FLAG','True_Flag'], axis=1)
        df4_full = df4.copy()

        # Run on df3 initially (before loop)
        df4_unary = find_unary(df4) # Why don't I need to put self on this code? 

        # df4 = drop_unary(df4)
        df4 = df4.drop(columns=df3_unary, axis=1)

        binary, non_binary = find_binary(df4)
        column_order = binary + non_binary # for something later in in the NN code

        #Normalize non_binary data
        df4_std = better_scaling(df4)

        df4_std = df4_std[df3_std.columns] #Global Inheritance

        # Only works around 1750 rows and 35 columns if I keep features F_PU11 & S_PU11 F_PU6 and S_PU6
        scores_pca4 = pca.transform(df4_std)
        PCA4 = pd.DataFrame(scores_pca4, index=df4.index)
        # print(PCA4)

        return PCA4
    
    
    def panel(self):
        dataframe = self.output()
        return pn.Param(self.param.list_select)
        # return pn.Column(self.param.std_threshold , f'processing  \'{self.file_selector}\' data file.')
        # return pn.pane.Markdown(f'processing  \'{self.file_selector}\' data file.')
    
    @param.output('file_data')
    def output(self):
        return self.get_data()

class Visualization(DataProcessing): 
    # file_data=param.DataFrame(default=pd.DataFrame(get_data('/Users/cconner/Desktop/CC_BATADAL/Data/sim4.csv')))
    # file_data=param.DataFrame()
    # file_data_copy=file_data
    file_data=param.DataFrame()
    x = param.Selector(default=15, objects=int_components)
    thread_Trigger = param.Boolean(default=True)
    std_threshold = param.Number(default = 3, bounds=(2.00, 5.00), precedence=0.5)
    iterations = param.Integer(default=0)
    
    # c = pn.widgets.CheckBoxGroup(name='Checkbox Group', value=components, options=components, inline=True)
    # c = param.ListSelector(objects=int_components)
    component_list = param.ListSelector(default=[1,3,5], objects=int_components, precedence=0.5)



    
    def update_data(self):  # function to be ran for continuous updating. 
        global stop_thread  # The function with inherit global variable data & stop_threadwhile not stop_thread:    # While stop_thread does not equate to FALSE
        # new_time = pd.to_datetime('now')
        # new_time = pd.to_datetime('now',utc=True)   # Get the current date_time..
        while not stop_thread: 
            new_file_data = get_data(self.file_selector)
            if  self.file_data.index[-1]  == new_file_data.index[-1]:
                return self.file_data
            else:
                last_time_record = len(self.file_data.index) - 1
                current_time_full_list = len(new_file_data.index)
                # current_time_record = len(new_file_data.index) - 1
                loc_of_copy_index = current_time_full_list - (current_time_full_list -last_time_record)

                # new_file_data.iloc[[loc_of_copy_index + 1]] #starting point of new row(s)
                to_append = new_file_data.iloc[(loc_of_copy_index + 1):] # Starting point to the last value.....
                self.file_data = pd.concat([self.file_data,to_append], ignore_index=True)
                self.iterations += 1
                time.sleep(1) 
    
    
    
    
    def call_back(self):  # function to be ran for continuous updating. 
        global stop_thread  # The function with inherit global variable data & stop_threadwhile not stop_thread:    # While stop_thread does not equate to FALSE
        while not self.thread_Trigger:
            new_file_data = get_data(self.file_selector) #'/Users/cconner/Desktop/CC_BATADAL/Data/sim4.csv'
            if  self.file_data.index[-1]  == new_file_data.index[-1]:
                # print("It is the same")
                # return file_data
                time.sleep(.5)
            else:
                # print("Not the same, updating")
                last_time_record = len(self.file_data.index) - 1
                current_time_full_list = len(new_file_data.index)
                # current_time_record = len(new_file_data.index) - 1
                loc_of_copy_index = current_time_full_list - (current_time_full_list -last_time_record)
                
                # new_file_data.iloc[[loc_of_copy_index + 1]] #starting point of new row(s)
                to_append = new_file_data.iloc[(loc_of_copy_index + 1):] # Starting point to the last value.....
                self.file_data = pd.concat([self.file_data,to_append], ignore_index=False)
                self.iterations += 1
                # return file_data
            time.sleep(.5)
    
    
    @param.depends('thread_Trigger', watch=True)
    def threading_start(self):
        if self.thread_Trigger == False:
            threading.Thread(target=self.call_back, daemon=True).start()
            # threading.Thread(target=self.update_data, daemon=True).start()
        else:  
            self.thread_Trigger = True

            
    @param.depends('x', 'iterations')
    def anomolies_detected_min_max(self):
        component = self.x
        string_x = str(self.x)
        column = self.file_data[component]
        column_max = PCA3[component].max()
        column_min = PCA3[component].min()
        comp_anomalies = (column > column_max) | (column < column_min)
        return f'For Component {string_x}, {len(column[comp_anomalies].index)} anomalies were detected at the following times: {column[comp_anomalies].index.tolist()}'
        
    @param.depends('x', 'iterations')
    def pca_component_min_max(self):
        component = self.x
        string_x = str(self.x)
        column = self.file_data[component]
        column_max = PCA3[component].max()
        column_min = PCA3[component].min()
        comp_anomalies = (column > column_max) | (column < column_min)
        layout = ((self.file_data.hvplot(y=string_x, title=(f'Component {string_x}'), label='Data', color='black')).opts( show_grid=True,bgcolor='LightGray')
                  * (self.file_data[comp_anomalies].hvplot(y=string_x, kind='scatter', color='red', label='Anomalies'))) * hv.HSpan(column_max, column_min, label='Threshhold')  
        
        return layout
    
 
    @param.depends('component_list','std_threshold', 'iterations')
    def pca_component_all_min_max(self):
        plots = []
        for x in self.component_list:
            # component = int(self.x)
            component = x
            string_x = str(x)
            column = self.file_data[component]
            column_max = PCA3[component].max()
            column_min = PCA3[component].min()
            comp_anomalies = (column > column_max) | (column < column_min)
            overlay = ((self.file_data.hvplot(y=string_x, title=(f'Component {string_x}'), label='Data', color='black')).opts( show_grid=True,bgcolor='LightGray')
                      * (self.file_data[comp_anomalies].hvplot(y=string_x, kind='scatter', color='red', label='Anomalies'))) * hv.HSpan(column_max, column_min, label='Threshhold')
            plots.append(overlay)
        
        return hv.Layout(plots).cols(2)
    
    @param.depends('component_list','std_threshold', 'iterations')
    def pca_component_all(self):
        plots = []
        for x in self.component_list:
            # component = int(self.x)
            component = x
            string_x = str(x)
            column = self.file_data[component]
            upper = column.mean() + self.std_threshold * column.std() #emperical 
            lower = column.mean() - self.std_threshold * column.std()
            comp_anomalies = (column > upper) | (column < lower)
            overlay = ((self.file_data.hvplot(y=string_x, title=(f'Component {string_x}'), label='Data', color='black')).opts( show_grid=True,bgcolor='LightGray')
                      * (self.file_data[comp_anomalies].hvplot(y=string_x, kind='scatter', color='red', label='Anomalies'))) * hv.HSpan(upper, lower, label='Threshhold')
            plots.append(overlay)
        
        return hv.Layout(plots).cols(2)
             
            
            
    @param.depends('x','std_threshold', 'iterations')
    def anomolies_detected_threshold(self):
        # component = int(self.x)
        component = self.x
        string_x = str(self.x)
        column = self.file_data[component]
        upper = column.mean() + self.std_threshold * column.std() #emperical 
        lower = column.mean() - self.std_threshold * column.std()
        comp_anomalies = (column > upper) | (column < lower)
        # return f'For Component {string_x}, anomalies detected at {self.file_data[comp_anomalies][component].index.tolist()}'
        # return f'For Component {string_x}, anomalies detected at {self.file_data[comp_anomalies][component].index}'
        # return f'For Component {string_x}, anomalies detected at {column[comp_anomalies].index}'
        return f'For Component {string_x}, {len(column[comp_anomalies].index)} anomalies were detected at the following times: {column[comp_anomalies].index.tolist()}'
        
    
    
    @param.depends('x','std_threshold', 'iterations')
    def pca_component_threshold(self):
        # component = int(self.x)
        component = self.x
        string_x = str(self.x)
        column = self.file_data[component]
        upper = column.mean() + self.std_threshold * column.std() #emperical 
        lower = column.mean() - self.std_threshold * column.std()
        comp_anomalies = (column > upper) | (column < lower)
        layout = ((self.file_data.hvplot(y=string_x, title=(f'Component {string_x}'), label='Data', color='black')).opts( show_grid=True,bgcolor='LightGray')
                  * (self.file_data[comp_anomalies].hvplot(y=string_x, kind='scatter', color='red', label='Anomalies'))) * hv.HSpan(upper, lower, label='Threshhold')
        
        return layout
    
    def view(self):
        if self.list_select == 'Min/Max':
            self.param.std_threshold.precedence = -1
            self.param.component_list.precedence = 0.5
            widget = pn.Param(self.param['component_list'], widgets={"component_list": pn.widgets.CheckBoxGroup})      
            return pn.Column(pn.WidgetBox(self.param.x,self.param.thread_Trigger), pn.Tabs(('Single Plot',pn.Column(self.pca_component_min_max, self.anomolies_detected_min_max)),
                                                                                           ('Multi Plot',pn.Row(self.pca_component_all_min_max, widget))))
        elif self.list_select == 'Threshold':
            # self.param.component_list.precedence = -1
            self.param.std_threshold.precedence = 0.5
            widget = pn.Param(self.param['component_list'], widgets={"component_list": pn.widgets.CheckBoxGroup})
            # return pn.Column(pn.WidgetBox(self.param.x, self.param.std_threshold, self.param.thread_Trigger),self.pca_component_threshold, self.anomolies_detected_threshold)
            return pn.Column(pn.WidgetBox(self.param.x,self.param.std_threshold, self.param.thread_Trigger), pn.Tabs(('Single Plot',pn.Column(self.pca_component_threshold, self.anomolies_detected_threshold)),
                                                                                           ('Multi Plot',pn.Row(self.pca_component_all, widget))))
        else:
            print('No')
    
    
    # @param.depends('x')
    # def anomalies(self)
    # @param.depends('iterations', watch=True)
    def panel(self):
        # periodic_cb = pn.state.add_periodic_callback(self.output, period=50, start=False)
        # return pn.Column(pn.Column(self.param.x, self.param.std_threshold, self.param.thread_Trigger),self.pca_component, self.anomolies_detected)
        return self.view()


Stages = pn.pipeline.Pipeline()
Stages.add_stage('Input', InputFile)
Stages.add_stage('Processing', DataProcessing)
Stages.add_stage('Visual', Visualization)

pn.template.FastListTemplate(
    site="Panel Application", title="Anomaly Detection",
    # theme="dark",
    theme_toggle=True,
    main=[Stages]
).servable()


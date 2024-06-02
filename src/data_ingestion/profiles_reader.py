import pandas as pd 
import os
def get_profiles() : 
    data = pd.read_csv('../../profile_format_standardization/final_data.csv')
    return data 

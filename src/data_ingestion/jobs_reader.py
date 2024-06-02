import pandas as pd 
import os
def get_jobs() : 
    data = pd.read_json('../../jobs_scraping/jobs_scraping/keejob.json') 
    data2 = pd.read_json('../../jobs_scraping/jobs_scraping/tanit_jobs.json')  
    df_merged = pd.concat([data, data2], ignore_index=True, sort=False)
    return df_merged 
print(get_jobs())
import os 
import sys
from src.data_ingestion.connector import get_connection
os.environ['PYSPARK_PYTHON'] = r'C:\Users\rayen\Desktop\programming\big_data\torch_env\Scripts\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\rayen\Desktop\programming\big_data\torch_env\Scripts\python.exe'
from pymongo import MongoClient
import pandas as pd
from pyspark.sql import SparkSession
import findspark
from pyspark.sql.functions import col, udf
import numpy as np
from fuzzywuzzy import fuzz
from pyspark.sql.types import FloatType


def get_data(spark) : 
    client = get_connection() 
    db = client["jobs_profiles_db"] 
    jobs_collection = db['jobs']
    profiles_collection = db['profiles']
    data = []
    data_profiles = []
    for document in jobs_collection.find({}):
        data.append(document)
    df = pd.DataFrame(data)
    for document in profiles_collection.find({}):
        data_profiles.append(document)
    profiles_df = pd.DataFrame(data_profiles)
    df=df.drop('_id',axis=1)
    profiles_df = profiles_df.drop('_id',axis=1) 
    spark_df = spark.createDataFrame(df)
    profiles_df_spark =spark.createDataFrame(profiles_df)
    return profiles_df_spark,spark_df

def initialize_session() : 
    findspark.init()
    spark = SparkSession.builder \
        .appName("ContentBasedFiltering") \
        .config("spark.pyspark.python", r"C:\Users\rayen\Desktop\programming\big_data\torch_env\Scripts\python.exe") \
        .config("spark.pyspark.driver.python", r"C:\Users\rayen\Desktop\programming\big_data\torch_env\Scripts\python.exe") \
        .config("spark.memory.offHeap.enabled","true") \
        .config("spark.memory.offHeap.size","10g") \
        .getOrCreate()
    return spark 
def location_similarity(profile_loc,job_loc,threshold=90) : 
    if job_loc :
        similarity_score = fuzz.ratio(profile_loc.lower(), job_loc.lower())
        return float(1) if similarity_score >= threshold else float(0)
    else:
        return float(1)
def level_similarity(profile_level, job_level):
    return float(1) if profile_level == job_level else float(0)
def experience_similarity( profile_exp,job_exp):
    profile_years = int(profile_exp / 365)
    if not profile_years : 
        return float(0) 
    if job_exp :  
        if job_exp[0] and job_exp[1] : 
            if job_exp[0] <= profile_years <= job_exp[1]:
                return float(1)
            elif profile_years > job_exp[1]:
                return float(1.2)
            else:
                return float(0)
        else : 
            return float(1)
    else : 
        return float(1)
def skills_similiarity(profile_skills, job_skills,job_description):
    resume_skills_set = set(map(str.lower, profile_skills))
    job_skills_set = set(map(str.lower, job_skills))
    common_skills = resume_skills_set & job_skills_set
    for skill in resume_skills_set.copy():
        if skill in job_description.lower() : 
            common_skills.add(skill)
            job_skills_set.add(skill)
        for job_skill in job_skills_set:
            if skill in job_skill or job_skill in skill:
                common_skills.add(job_skill)
    if len(job_skills_set ) : 
        return len(common_skills) / len(job_skills_set)
    return 1
def get_jobs_similiarity(profile,jobs_df) : 
    threshold = 80
    """
    returns the jobs with their respective similiarities with the given profile
    """
    location_udf_profileside= udf(lambda job_loc: location_similarity(profile['location'], job_loc,threshold), FloatType())
    level_udf_profileside = udf(lambda job_level : level_similarity(profile['max_level'],job_level),FloatType())
    experience_udf_profileside = udf(lambda job_experience : experience_similarity(profile['experience'],job_experience),FloatType())
    skills_udf_profileside = udf(lambda job_skills,job_description : skills_similiarity(profile['skills'],job_skills,job_description),FloatType())
    weights = {"skills_sim": 0.6, "location_sim": 0.2, "level_sim": 0.1, "experience_sim": 0.1}

    job_df_with_similarity = jobs_df.withColumn(
        "level_sim", level_udf_profileside(jobs_df["niveau"])
    ).withColumn(
        "location_sim", location_udf_profileside(jobs_df["location"])
    ).withColumn(
        "experience_sim", experience_udf_profileside(jobs_df["experience"])
    ).withColumn(
        "skills_sim", skills_udf_profileside(jobs_df["skills"], jobs_df["description"])
    )
    job_df_with_similarity = job_df_with_similarity.withColumn(
        "overall_similarity", 
        sum(col(similarity) * weights[similarity] for similarity in weights)
    )
    return job_df_with_similarity 
def get_profiles_similiarity(job,profiles) : 
    threshold = 80
    """
    return the profiles with their similarities given a job
    """
    location_udf_profileside= udf(lambda profile_loc: location_similarity(profile_loc, job['location'],threshold), FloatType())
    level_udf_profileside = udf(lambda profile_level : level_similarity(profile_level,job['niveau']),FloatType())
    experience_udf_profileside = udf(lambda profile_experience : experience_similarity(profile_experience,job['experience']),FloatType())
    skills_udf_profileside = udf(lambda profile_skills : skills_similiarity(profile_skills,job['skills'],job['description']),FloatType())
    weights = {"skills_sim": 0.6, "location_sim": 0.2, "level_sim": 0.1, "experience_sim": 0.1}

    profile_df_with_similarity = profiles.withColumn(
        "level_sim", level_udf_profileside(profiles["max_level"])
    ).withColumn(
        "location_sim", location_udf_profileside(profiles["location"])
    ).withColumn(
        "experience_sim", experience_udf_profileside(profiles["days_experience"])
    ).withColumn(
        "skills_sim", skills_udf_profileside(profiles["skills"])
    )
    profile_df_with_similarity = profile_df_with_similarity.withColumn(
        "overall_similarity", 
        sum(col(similarity) * weights[similarity] for similarity in weights)
    )
    return profile_df_with_similarity 


spark = initialize_session()
profile = {"skills":["Python", "SQL", "Java", "docker",'CI CD'],'location':'tunis',"experience":600,'max_level':'Bac + 3'}
job = {"skills":["Python", "SQL", "Java", "docker",'CI CD'],'location':'tunis',"experience":[0,1],'niveau':'Bac + 3','description':'Integration Objects is a global leader in industrial digital transformation solutions including: Industrial Iot (IIoT), Cybersecurity, Data Analytics, Big Data, Process Control and Automation Systems. To strengthen its team, Integration Objects is recruiting: 1- Full Project Cycle Management: Lead the project journey, from the collection of initial requirements to the meticulous definition of execution plans. 2- Progress Monitoring: Document and closely monitor the progress of the project using Key Performance Indicators (KPIs) and Success Indicators. 3- Risk Assessment and Mitigation: Conduct in-depth project risk assessments and develop effective mitigation strategies to ensure project success. 4- Communication with Stakeholders: Maintain transparent communication with all project stakeholders and ensure that their needs are met throughout the project life cycle. 5- Resource Optimization: Effectively manage project resources to maximize productivity and minimize waste. 6- Budget and Schedule Management: Continuously update and manage the project schedule and budget to maintain alignment with project objectives. : -You hold a diploma in Industrial, Automatic, Electrical, or Instrumentation Engineering, or an equivalent qualification. -You have at least 5 years of experience or more in a similar position. -Strong knowledge of PLC, DCS/SCADA systems and telecommunications development. -Experience in FAT, SAT, operational verification, automation project start-up. -Good level of French in English (oral and written) -Strong organizational skills, supported by a good spirit of analysis and synthesis, allowing a high degree of autonomy'}
profiles_df, jobs_df = get_data(spark) 

profiles_with_similarities = get_profiles_similiarity(job,profiles_df)
# Select both the name and overall_similarity columns
top_similarity_scores = profiles_with_similarities.orderBy(col("overall_similarity").desc()).select("name", "overall_similarity")

#jobs_with_similarities = get_jobs_similiarity(profile,jobs_df)
#top_similarity_scores = jobs_with_similarities.orderBy(col("overall_similarity").desc()).select("overall_similarity")
top_similarity_scores.show()
spark.stop()




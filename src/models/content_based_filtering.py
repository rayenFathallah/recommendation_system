import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing

profiles = pd.read_json('output.json',orient='records')  
jobs = pd.read_json('new_jobs.json',orient='records')   
from fuzzywuzzy import fuzz

def location_similiary(profile_loc, job_loc, threshold):
    
    similarity_score = fuzz.ratio(profile_loc.lower(), job_loc.lower())
    if similarity_score >= threshold : 
        return 1 
    return 0  
def level_similarity(profile_level,job_level) : 
    if job_level==profile_level : 
        return 1 
    return 0 
def experience_similiarity(job_exp,profile_exp) : 
    profile_years = int(profile_exp / 365) 
    if profile_years > job_exp[0] and profile_years < job_exp[1]  : 
        return 1 
    elif profile_years > job_exp[0] and profile_years > job_exp[1] : 
        return 1.2 
    else : 
        return 1 

def skills_similiarity(job_skills, profile_skills):
    # Convert lists to sets to remove duplicates and make order irrelevant
    job_skills_set = set(job_skills)
    profile_skills_set = set(profile_skills)
    
    # Compute intersection of skills
    intersecting_skills = job_skills_set.intersection(profile_skills_set)
    
    if not intersecting_skills:
        return 0.0  # If no intersecting skills, return 0 similarity
    
    # Create binary vectors representing skills
    job_skill_vector = np.array([1 if skill in job_skills_set else 0 for skill in job_skills_set])
    profile_skill_vector = np.array([1 if skill in intersecting_skills else 0 for skill in job_skills_set])
    
    # Reshape vectors for cosine similarity calculation
    job_skill_vector = job_skill_vector.reshape(1, -1)
    profile_skill_vector = profile_skill_vector.reshape(1, -1)
    
    # Compute cosine similarity between profile and job skills
    similarity = cosine_similarity(job_skill_vector, profile_skill_vector)[0][0]
    
    return similarity

def extract_similarity(job,profile,weights) : 
    skills_sem = skills_similiarity(job['skills'],profile['skills']) 
    location_sem = location_similiary(profile['location'],job['location'],0.9)
    exp_sim = experience_similiarity(job['experience'],profile['days_experience']) 
    level_sim = level_similarity(profile['max_level'],job['education']) 
    final_similarity = skills_sem * weights['skills'] + location_sem * weights['location'] + exp_sim * weights['experience'] + level_sim * weights['level'] 
    return final_similarity
def calculate_similarity_for_index(profile, job, idx):
    
    weights = {'skills' : 0.6,'location':0.1,'experience':0.2,'level':0.1}
    similarity_score = extract_similarity(job, profile,weights)
    return similarity_score, idx
def calculate_similarities_parallel(profiles, job):

    with multiprocessing.Pool() as pool:
        results = pool.starmap(calculate_similarity_for_index, [(profiles, job, idx) for idx in range(len(profiles))])
    return results
def calculate_similarities(profiles, job):
    
    similarities = []
    for idx, profile in enumerate(profiles):
            similarity_score = calculate_similarity_for_index(profile, job,idx)
            similarities.append((similarity_score, idx))
    return similarities


job = jobs.to_dict(orient='records')[1] 
profs = profiles.to_dict(orient='records')
results = calculate_similarities(profs,job)
print(results[:10])

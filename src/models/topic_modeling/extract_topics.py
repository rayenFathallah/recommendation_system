import sys
sys.path.append("C:/Users/rayen/Desktop/programming/big_data/src/models/topic_modeling") 
import os

def get_topics(elems,loaded_model) : 
    topic_model = loaded_model
    texts = [job.get("description") for job in elems]  
    all_topics = topic_model.transform(texts)
    print('length of elements' ,len(elems))
    print('length of topics : ',len(all_topics[0]))
    for job, topics in zip(elems, all_topics[0]):
        job["topic"] = topics

    return elems



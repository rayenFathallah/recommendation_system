import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk import ngrams
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import torch
import sys
#import spacy
from torch.utils.data import Dataset, DataLoader
from torch import nn
from deep_translator import GoogleTranslator
from langdetect import detect
import time
from concurrent.futures import ThreadPoolExecutor
import transformers
from transformers import pipeline, AutoTokenizer, AutoModel

#nltk.download('punkt') run this once  

class EscoDataset(Dataset):
    def __init__(self, df, skill_col, backbone):
        texts = df
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.texts = texts[skill_col].values.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        res = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=20
        )
        return {k:v[0] for k,v in res.items()}

    
class ClsPool(nn.Module):
    def forward(self, x):
        # batch * num_tokens * num_embedding
        return x[:, 0, :]    

    
class BertModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        self.backbone_name = backbone
        self.backbone = AutoModel.from_pretrained(backbone)
        self.pool = ClsPool()
    
    def forward(self, x):
        x = self.backbone(**x)["last_hidden_state"]
        x = self.pool(x)
        
        return x
class final_model() : 
    def __init__(self) :   
        self.emb_label = 'jobbert'
        model_path = os.path.abspath('./src/models/bert_model_instance.pth')

        try : 
            self.model = torch.load("bert_model_instance.pth")
        except : 
            print("the model path is " ,model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = 'jjzha/jobbert-base-cased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self.esco_df = pd.read_csv('skills_en.csv')
        self.esco_df['label_cleaned'] = self.esco_df['preferredLabel'].apply(lambda x: re.sub(r'\([^)]*\)', '', x).strip())
        self.esco_df['word_cnt'] = self.esco_df['label_cleaned'].apply(lambda x: len(str(x).split()))
        self.esco_df = pd.DataFrame(self.esco_df, columns=['label_cleaned', 'altLabels', 'word_cnt'])
        embs = []
        ds = EscoDataset(self.esco_df, 'label_cleaned', self.backbone)
        dl = DataLoader(ds, shuffle=False, batch_size=32)
        self.model.to(self.device) 
        
        with torch.no_grad():
            for i, x in enumerate(dl):
                x = {k:v.to(self.device) for k, v in x.items()}
                out = self.model(x)
                embs.extend(out.detach().cpu())
        
        self.esco_df[self.emb_label] = embs


    def get_sentences(self,job):
        """
        Given a raw html job description, parse it into sentences
        by using nltk's sentence tokenization + new line splitting, this can also accept raw text not only html text
        """
        soup = BeautifulSoup(job, 'html.parser')
        # Found some ads using unicode bullet points
        for p in soup.find_all('p'):
            p.string = p.get_text().replace("•", "")
        text = soup.get_text()
        st = sent_tokenize(text)
        sentences = []
        for sent in st:
            sentences.extend([x for x in sent.split('\n') if x !=''])
        return sentences
    def get_embedding(self,x):
        x = self.tokenizer(x, return_tensors='pt')
        x = {k:v.to(self.device) for k, v in x.items()}
        return self.model(x).detach().cpu()
    def compute_similarity_opt(self,emb_vec, emb_type):
        """
        Compute vector similarity for a given vec and all the ESCO skills embeddings
        by constructing a matrix from ESCO embeddings to process it faster.
        Return the ESCO skill id with max similarity
        """
        esco_embs = [x for x in self.esco_df[emb_type]]
        esco_vectors = torch.stack(esco_embs)
        # Normalize the stacked embeddings and the input vector
        norm_esco_vectors = torch.nn.functional.normalize(esco_vectors, p=2, dim=1)
        norm_emb_vec = torch.nn.functional.normalize(emb_vec.T, p=2, dim=0)
        # Compute cosine similarities
        cos_similarities = torch.matmul(norm_esco_vectors, norm_emb_vec)
        # Return max similarity and esco skill index
        sim, idx = torch.max(cos_similarities, dim=0)
        return idx.item(), sim.item()

    def compute_similarity_mat(self,emb_mat, emb_type):
        esco_embs = [x for x in self.esco_df[emb_type]]
        esco_vectors = torch.stack(esco_embs)
        emb_vectors = torch.stack(emb_mat)
        # Normalize the stacked embeddings and the input vectors
        norm_esco_vectors = torch.nn.functional.normalize(esco_vectors, p=2, dim=1)
        norm_emb_vecs = torch.nn.functional.normalize(emb_vectors.T, p=2, dim=0)
        # Compute cosine similarities
        cos_similarities = torch.matmul(norm_esco_vectors, norm_emb_vecs)
        # Return max similarity and esco skill index
        max_similarities, max_indices = torch.max(cos_similarities, dim=0)
        return max_indices.numpy(), max_similarities.numpy()
    def process_sentence(self,sent):
        emb = self.get_embedding(sent)
        return self.compute_similarity_opt(emb, self.emb_label)

    def get_classifiers(self,mtype):
        if mtype == "jobbert":
            token_skill_classifier = pipeline(model="jjzha/jobbert_skill_extraction", aggregation_strategy="first", device=self.device)
            token_knowledge_classifier = pipeline(model="jjzha/jobbert_knowledge_extraction", aggregation_strategy="first", device=self.device)
        elif mtype == "xlmr":        
            token_skill_classifier = pipeline(model="jjzha/escoxlmr_skill_extraction", aggregation_strategy="first", device=self.device)
            token_knowledge_classifier = pipeline(model="jjzha/escoxlmr_knowledge_extraction", aggregation_strategy="first", device=self.device)
        else:
            raise Exception("Unknown model name provided")
        return token_skill_classifier, token_knowledge_classifier


    def extract_skills(self,job, token_skill_classifier, token_knowledge_classifier, out_treshold=.8, sim_threshold=.8):
        """
        Function that processes outputs from pre-trained, ready to use models
        that detect skills as a token classification task. There are two thresholds,
        out_threshold for filtering model outputs and sim_threshold for filtering
        based on vector similarity with ESCO skills
        """     
        sentences = self.get_sentences(job)
        pred_labels = []
        res = []
        skill_embs = []
        skill_texts = []
        for sent in sentences:
            skills = self.ner(sent, token_skill_classifier, token_knowledge_classifier)
            for entity in skills['entities']:
                text = entity['word']
                if entity['score'] > out_treshold:
                    skill_embs.append(self.get_embedding(text).squeeze())
                    skill_texts.append(text)
                    
        idxs, sims = self.compute_similarity_mat(skill_embs, self.emb_label)
        for i in range(len(idxs)):
            if sims[i] > sim_threshold:
                pred_labels.append(idxs[i])
                res.append((skill_texts[i], self.esco_df.iloc[idxs[i]]['label_cleaned'], sims[i]))
        
        return pred_labels, res


    def aggregate_span(self,results):
        new_results = []
        current_result = results[0]

        for result in results[1:]:
            if result["start"] == current_result["end"] + 1:
                current_result["word"] += " " + result["word"]
                current_result["end"] = result["end"]
            else:
                new_results.append(current_result)
                current_result = result

        new_results.append(current_result)

        return new_results


    def ner(self,text, token_skill_classifier, token_knowledge_classifier):
        output_skills = token_skill_classifier(text)
        for result in output_skills:
            if result.get("entity_group"):
                result["entity"] = "Skill"
                del result["entity_group"]

        output_knowledge = token_knowledge_classifier(text)
        for result in output_knowledge:
            if result.get("entity_group"):
                result["entity"] = "Knowledge"
                del result["entity_group"]

        if len(output_skills) > 0:
            output_skills = self.aggregate_span(output_skills)
        if len(output_knowledge) > 0:
            output_knowledge = self.aggregate_span(output_knowledge)
        
        skills = []
        skills.extend(output_skills)
        skills.extend(output_knowledge)
        return {"text": text, "entities": skills}


def translate_text(resume_text) :
    try :
        if resume_text !="" : 
            lang = detect(resume_text)
            if lang == 'fr' :
                final_text=list()
                for i in range(0, len(resume_text), 50000):
                    translation = GoogleTranslator(source='auto', target='en').translate(resume_text[i:i+50000], dest='en')
                    final_text.append(translation)
                return ' '.join(final_text) 
            else : 
                return resume_text
        else : 
            return ""
    except Exception as e:
      # Catch any other exceptions and log error message
        return resume_text

JOBS_FP = 'final_jobs.json'
df = pd.read_json(JOBS_FP)
#df['description'] = df['description'].apply(translate_text) 
#job_sample = df.iloc[3]


def extract_skills_all(jobs) : 
    test_model = final_model()
    tsc, tkc = test_model.get_classifiers("jobbert") 
    with ThreadPoolExecutor(max_workers=16) as executor:

        for job in jobs : 
            future = executor.submit(process_job,translate_text(job['description']),tsc,tkc,test_model)
            job['skills'] = future.result()
    return jobs
def extract_skills_job(job)  : 
    test_model = final_model()
    tsc, tkc = test_model.get_classifiers("jobbert") 
    job['skills'] = process_job(translate_text(job['description']),tsc,tkc,test_model) 
    return job
def process_job(job,tsc,tkc,test_model) : 
    if job=='' : 
        return []
    final_skills = set()
    try : 
        _, res = test_model.extract_skills(job, tsc, tkc) 
        for r in res: 
            if len(r) > 1 : 
                final_skills.add(r[1]) 
        return final_skills
    except : 
        print("a problem has occured here")
        print(job)
        return []

start = time.time()
dict_list = df.to_dict(orient='records')
new_jobs = extract_skills_all(dict_list)
new_jobs_df = pd.DataFrame(new_jobs)
new_jobs_df.to_json('new_jobs.json',orient='records')
end = time.time() 
print(end-start)










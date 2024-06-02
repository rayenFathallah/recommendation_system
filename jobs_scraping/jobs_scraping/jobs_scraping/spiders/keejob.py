import scrapy
from bs4 import BeautifulSoup
from datetime import datetime
import requests
from scrapy.http import TextResponse
from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords
import pymongo
from deep_translator import GoogleTranslator
from langdetect import detect
from dotenv import load_dotenv
import os
#from src.models.skill_extraction_model import extract_skills_all
load_dotenv()

class KeejobSpider(scrapy.Spider):
    name = "keejob"
    allowed_domains = ["keejob.com"]
    start_urls = ["http://www.keejob.com/offres-emploi/"]
    all_elements = []
    gouvs = ['sfax',
 'ariana',
 'kef',
 'kairouan',
 'le kef',
 'medenine',
 'zaghouan',
 'gafsa',
 'siliana',
 'jendouba',
 'béja',
 'gabès',
 'nabeul',
 'mahdia',
 'kébili',
 'sousse',
 'tataouine',
 'manouba',
 'gabes',
 'tozeur',
 'médnine',
 'kasserine',
 'sidi Bouzid',
 'ben Arous',
 'kebili',
 'tunis',
 'bizerte',
 'monastir']
    def parse(self, response):
        all_elems =  self.parse_page(response.url)
        mongo_uri = os.getenv("mongoURI")
        client = pymongo.MongoClient(mongo_uri)
        db = client.recommendation_jobs 
        collection = db["jobs"]
        for elem in all_elems : 
            yield elem

    def parse_page(self, url,all_jobs_result=[]):
        html_content = self.get_html_content(url)
        response = TextResponse(url=url, body=html_content, encoding='utf-8')
        container = response.css('#loop-container')
        all_jobs = container.css('.clearfix')
        with ThreadPoolExecutor(max_workers=16) as executor:
            for job in all_jobs:
                href = job.css('.span8 a::attr(href)').get()
                future = executor.submit(self.parse_job,response.urljoin(href))
                all_jobs_result.append(future.result())
        #print(len(all_jobs_result))
        next_page = self.get_next_page(response)
        if next_page:
            self.parse_page(next_page,all_jobs_result)
        #else : 
        print(len(all_jobs_result))
        return all_jobs_result

    def parse_job(self, url):
            html_content = self.get_html_content(url)
            response = TextResponse(url=url, body=html_content, encoding='utf-8')
            span_content = response.css('.span9.content')
            job = {}
            a_tag = span_content.css('a')
            if a_tag : 
                job['company_name'] = a_tag.css('a::text').get()
                all_metas = response.css('.meta') 
                for meta in all_metas : 
                    name = meta.css('b::text').get() 
                    html_text = meta.get()
                    if name : 
                        if name == 'Référence:' : 
                            job['reference'] = self.extract_text(html_text) 
                        elif name == 'Publiée le:': 
                            extracted_text = self.extract_text(html_text)
                            day, month, year = extracted_text.split()
                            month = self.french_to_english_month(month)
                            job['date'] = datetime.strptime(f"{day} {month} {year}", "%d %B %Y") 
                        elif name == 'Lieu de travail:'  : 
                            
                            location = self.extract_text(html_text)
                            job['location'] = self.extract_location(location)
                        elif name == 'Expérience:' : 
                            experience = self.extract_text(html_text)
                            job['experience'] = self.extract_experience_keejob(experience)
                        elif name == "Étude:" : 
                            job['education']  = self.extract_text(html_text) 
                        elif name =='Langues' : 
                            job['languages'] = self.extract_text(html_text) 
                            
                description_wrapper = response.css('.block_a.span12.no-margin-left')
                spans = description_wrapper.css('span::text').extract()
                lis = description_wrapper.css('li::text').extract()
                pars = description_wrapper.css('p::text').extract()
                final_list = spans + lis + pars
                str_desc = ' '.join(final_list)
                cleaned_description = self.clean_description(str_desc)
                job['description'] = cleaned_description
                job['url'] = response.url 
                job['website'] = 'keejob'
                job["description_eng"] = self.detect_lang(job['description'])
                #job['skills'] = self.extract_skills(cleaned_description) 

                return job

    def get_html_content(self, url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.content
            else:
                self.logger.error(f"Failed to fetch URL: {url}")
                return None
        except Exception as e:
            self.logger.error(f"An error occurred while fetching URL: {url}, Error: {e}")
            return None

    def french_to_english_month(self, month):
        months = {
            'janvier': 'January',
            'février': 'February',
            'mars': 'March',
            'avril': 'April',
            'mai': 'May',
            'juin': 'June',
            'juillet': 'July',
            'août': 'August',
            'septembre': 'September',
            'octobre': 'October',
            'novembre': 'November',
            'décembre': 'December'
        }
        return months.get(month.lower(), month)

    def extract_text(self, html_text):
        html_text = html_text.replace('\n', '').replace('\t', '').strip()
        soup = BeautifulSoup(html_text, 'html.parser')
        meta_div = soup.find('div', class_='meta')
        extracted_text = meta_div.find('br').next_sibling.strip()
        return extracted_text.replace('  ', ' ').strip()

    def clean_description(self, text):
        stop_words = set(stopwords.words('english'))
        text = text.replace('\xa0', ' ')
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_text = ' '.join(filtered_words)
        return cleaned_text

    def get_next_page(self, response):
        next_page = response.css('[aria-label="Next Page"]::attr(href)').get()
        if next_page:
            return response.urljoin(next_page)
        return None
    def extract_experience_keejob(self,exp) : 
        association_dict = {'Aucune expérience' : [0,0],"Moins d'un an"  : [0,1], "Moins de deux ans"  : [0,2],"Moins de trois ans"  : [0,3],
                            "Entre 1 et 2 ans" : [1,2], 'Entre 2 et 5 ans' : [2,5],'Entre 1 et 3 ans' : [1,3], "Entre 5 et 10 ans" : [5,10],
                            'Plus que 10 ans' : [10,float('inf')]}
                            
        if exp in association_dict.keys() : 
            return association_dict[exp] 
        return [0,float('inf')] 
    def extract_location(self,loc) : 
        for elem in self.gouvs : 
            if elem.lower() in loc : 
                return elem
    
    def detect_lang(self,resume_text) :
        try :
            lang = detect(resume_text)
            if lang == 'fr' :
                final_text=list()
                for i in range(0, len(resume_text), 50000):
                    translation = GoogleTranslator(source='auto', target='en').translate(resume_text[i:i+50000], dest='en')
                    final_text.append(translation)
                return ' '.join(final_text) 
            else : 
                return resume_text
        except Exception as e:
            return resume_text     
        
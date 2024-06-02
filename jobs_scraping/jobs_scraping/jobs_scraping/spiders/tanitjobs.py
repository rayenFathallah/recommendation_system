import scrapy
import re
#from concurrent.futures import ThreadPoolExecutor
from nltk.corpus import stopwords

class tanitjobsSpider(scrapy.Spider):
    name = "tanitjobs"
    allowed_domains = ["www.tanitjobs.com"]
    start_urls = ["https://www.tanitjobs.com/jobs/?searchId=1712437326.0911&action=search"]
    qualifs = {'ingénieur':'Bac + 5',
    'licence, bac + 3':'Bac + 3',
    'dut, bts, bac + 2':"Bac + 2",
    'dess, dea, master, bac + 5, grandes ecoles':"Bac + 5",
    'maîtrise, iep, iup, bac + 4':"Bac + 4",
    'bac non validé':"Secondaire",
    'diplôme non validé':"Bac + 3",
    'lycée, niveau bac':"Secondaire",
    'bac professionnel, bep, cap':"Bac",
    'xpert, recherche':"Expert, Recherche",
    'etudiant':"Bac + 2",
    'doctorat, phd':"Doctorat",
    'médecin généraliste':"Expert, Recherche",
    'médecin spécialiste':"Expert, Recherche"}
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
        all_jobs = response.css('article') 
        current_page = response.css('#list_current_page b::text').get()
        last_page_list = response.css('#list_nav')[0].css('span a::text')[-1].get()              
        for job in all_jobs : 
            href = job.css('.link::attr(href)')[0].get()
            location = job.css('.listing-item__info--item-location::text').get().strip().lower()
            yield scrapy.Request(href, callback=self.parse_level2,meta={"location":location})  
        if int(current_page) != int(last_page_list) +1 :
            print("going to :" ,current_page+'1')
            current_page_span = response.css('#list_current_page')[0]
            following_span = current_page_span.xpath('following-sibling::span[1]')
            href = following_span.css('a::attr(href)').get()
            yield response.follow(href,callback = self.parse)    
    def parse_level2(self,response) : 
        job = dict() 
        location = response.meta.get('location')
        job['location'] = self.extract_location(location)
        job['title'] = response.css('.details-header__title::text').get().strip().lower() 
        job['company'] = response.css('.listing-item__info--item-company::text').get().strip().lower()
        details = response.css('.infos_job_details dl')
        for detail in details : 
            detail_type = detail.css('dt::text').get()
            if detail_type == 'Experience :' : 
                experience = detail.css('dd::text').get().strip().lower() 
                job['experience'] = self.extract_experience_tanit(experience)
            elif detail_type == "Niveau d'étude :" : 
                job['education'] = detail.css('dd::text').get().strip().lower() 
                job['niveau'] = self.extract_niveau(job['education'])
            elif detail_type == "Langue :" : 
                job['languages'] = detail.css('dd::text').get().strip().lower() 
        description = ''.join(response.css('p::text').extract()) 
        job['description'] = self.clean_description(description) 
        job['url'] = response.url 
        job['website'] = 'tanitjobs'
        yield job
    def clean_description(self, text):
        stop_words = set(stopwords.words('english'))
        text = text.replace('\xa0', ' ')
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_text = ' '.join(filtered_words)
        return cleaned_text

    def extract_experience_tanit(self,exp) : 
        association_dict = {'débutant' : [0,0],'0 à 1 an'  : [0,1], "Moins de deux ans"  : [0,2],"Moins de trois ans"  : [0,3],
                            '1 à 3 ans' : [1,3], '3 à 5 ans' : [3,5],'5 à 10 ans' : [5,10], "Entre 5 et 10 ans" : [5,10],
                            'plus 10 ans' : [10,float('inf')]}
                            
        if exp in association_dict.keys() : 
            return association_dict[exp] 
        return [0,float('inf')] 

    def extract_location(self,loc) : 
        for elem in self.gouvs : 
            if elem.lower() in loc : 
                return elem
    def extract_niveau(self,edu) : 
        if edu in self.qualifs.keys() : 
            return self.qualifs[edu] 
        return None      
            





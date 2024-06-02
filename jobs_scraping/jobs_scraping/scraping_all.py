import sys

sys.path.append("C:/Users/rayen/Desktop/programming/big_data")  
from jobs_scraping.jobs_scraping.jobs_scraping.spiders import tanitjobs,keejob
from scrapy.crawler import CrawlerProcess 
from scrapy.utils.project import get_project_settings 
import pymongo
import scrapy
from dotenv import load_dotenv
import os
import datetime 
import pandas as pd
from scrapy.utils.reactor import install_reactor
from src.models.skill_extraction_model import extract_skills_all
from src.models.topic_modeling.extract_topics import get_topics
load_dotenv()

def main():
    spiders = [tanitjobs.tanitjobsSpider,keejob.KeejobSpider] 

    #mongo_uri = os.getenv("mongoURI")
    #client = pymongo.MongoClient(mongo_uri)
    #db = client.products  # Modify based on your database name
    #collection = db["all_products_final"]

    settings = get_project_settings()
    process = CrawlerProcess(settings)

    all_results = []
    """
    def item_scraped(item, response, spider):
        print('Item scraped:', item)
        item['date'] = datetime.datetime.now()
        all_results.append(item)
        return item

    for spider_class in spiders:
        crawler = process.create_crawler(spider_class)
        crawler.signals.connect(item_scraped, signal=scrapy.signals.item_scraped)  # Connect to crawler instance
        process.crawl(crawler)  # Crawl the crawler instance

    process.start()
    jobs_with_skills = extract_skills_all(all_results) 
    final_jobs = get_topics(jobs_with_skills)
    print(final_jobs)
    """
    jobs_data = pd.read_json('final_jobs.json')
    jobs_with_skills = extract_skills_all(jobs_data) 


    #scoop_prods = main_fun_scoop()
    #all_results = all_results + scoop_prods
    #if len(all_results):
    #    collection.insert_many(all_results)

    #client.close()  
    
if __name__ == "__main__":
    main()
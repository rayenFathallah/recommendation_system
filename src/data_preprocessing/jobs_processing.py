from deep_translator import GoogleTranslator
from langdetect import detect
from src.logger import logging
def translate_text(resume_text) :
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
      # Catch any other exceptions and log error message
        logging.error("Error while detecting the language: " + str(e))
        return resume_text
"""
Instructions:
- open anaconda prompt as administrator
- naviagate inside the unzipped folder
- pip install -r requirements.txt
- python -m spacy download en_core_web_lg
- python ExtractiveSummarizer.py

Parameters:
All the parameters are set in config.py
Test parameters from line 35
"""


from PaperTest.tester import PaperTest
import nltk
nltk.download('stopwords')

tester = PaperTest()
print("PaperTest class built")
tester.TestPaper()      


        

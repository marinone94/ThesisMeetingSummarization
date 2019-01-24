This repository contains codes, corpora and report of my Master Thesis, where new features are introduced and evaluated, extending the current unsupervised extractive summarizer state-of-the-art. It is publicly available to reproduce my results and have a baseline to work on.

This code is designed to follow a complete pipeline, starting from transcript and reference text files, already extracted from the original corpus. 

You can train new topic models using the related package. 
In this case, set the correct folders and/or filneames in the config file.

All the parameters are hard-coded in cofig.py and anywhere else.
Consider that it is not 100% robust to unusual parameters.

All the reference to others' works can be found in the report refereces section.

Instructions:
- download and unzip the repository
- install anaconda wiht python 3.6
- open anaconda prompt as administrator
- naviagate inside the unzipped folder                                                   (.ThesisMeetingSummarization\ExtractiveSummarizer)
- pip install -r requirements.txt
- python -m spacy download en_core_web_lg
- python test_code.py

Parameters:
- All the parameters are set in config.py
- Test parameters from line 33

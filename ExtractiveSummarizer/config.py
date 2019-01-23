import spacy
import numpy as np
from nltk.corpus import stopwords


class Config(object):
    """ All the parameters and the other configurations are defined here
        No hard-coded configurations in other files
    """
    def __init__(self):
        #stopwords and punctuation
        self.stopwords      = stopwords.words('english')
        self.newStopwords   = ["um","uhm", "mm", "mmm", "yeah", "ehm", "mmh", "uhmm", "ah", "'m", "'re", "'s", "'ve", "hmm", "mm-hmm", "uh", "blah", "bah", "okay", "ok"]
        self.puncList       = ['.',',',':',';','!','?', '*', '+', '-', '_', '/', '<', '>', '@', '\\', '$' ,'€', '&', '£']
        self.alphabet       = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.pronLemma      = '-PRON-'
        self.ExtendStopwords()
        #load spacy - use 'en_core_web_md' or 'en_core_web_sm' for medium or small model size
        self.nlp            = spacy.load('en_core_web_lg') 
        #paths and filenames
        self.transcriptPath = r'.\Datasets\AMI\Transcripts\\'
        self.referencePath  = r'.\Datasets\AMI\References\\'
        self.histogramsPath = r'.\Datasets\AMI\Histograms\\'
        self.topicModelPath = r'.\Datasets\TopicModels\\'
        self.testResultPath = r'.\Datasets\AMI\Results\\'
        self.lpAddress      = r'.\Datasets\Optimization\opt_spacy.lp'
        self.histogramsFile = 'spacy_histo.npy'
        self.wordsFile      = 'spacy_single_words.txt'
        self.histoFolder    = 'histos\\'
        self.wordsFolder    = 'words\\'
        self.avgResultFile  = 'avg.npy'
        #topic dataset 1
        self.topicDict1     = ''.join([self.topicModelPath, 'meet_doc_dictionary.gensim'])
        self.topicCorpus1   = ''.join([self.topicModelPath, 'meet_doc_corpus.pkl'])
        self.topicModel1    = ''.join([self.topicModelPath, 'meet_doc_model.gensim'])
        self.topicDocs1     = ''.join([self.topicModelPath, 'tokens_topic_model_meet_doc.txt']) 
        #topic dataset 2
        self.topicDict2     = ''.join([self.topicModelPath, 'meet_doc_dictionary_bbc.gensim'])
        self.topicCorpus2   = ''.join([self.topicModelPath, 'meet_doc_corpus_bbc.pkl'])
        self.topicModel2    = ''.join([self.topicModelPath, 'meet_doc_model_bbc.gensim'])
        self.topicDocs2     = ''.join([self.topicModelPath, 'tokens_topic_model_meet_doc_bbc.txt']) 
        #topic dataset 3
        self.topicDict3     = ''.join([self.topicModelPath, 'meet_doc_dictionary_bbc_gensim_dataset.gensim'])
        self.topicCorpus3   = ''.join([self.topicModelPath, 'meet_doc_corpus_bbc_gensim_dataset.pkl'])
        self.topicModel3    = ''.join([self.topicModelPath, 'meet_doc_model_bbc_gensim_dataset.gensim'])
        self.topicDocs3     = ''.join([self.topicModelPath, 'tokens_topic_model_meet_doc_bbc_gensim_dataset.txt']) 
        #flags - ONLY ONE TOPIC MODEL SET CAN BE SET TO TRUE - default config with topicModelSet2 = True
        self.loadHistograms = True
        self.loadTopicModel = True
        self.topicModelSet1 = False     #topic model trained on AMI corpus
        self.topicModelSet2 = True      #topic model trained on AMI corpus + BBC news
        self.topicModelSet3 = False     #topic model trained on AMI corpus + BBC news + dataset.csv (https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/dataset.csv)
        self.CheckTopicModelSet()
        #POS
        self.allowedPOS     = ['N', 'NN', 'NNP', 'NNPS', 'NNS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        #NER
        self.listNER        = ['GPE', 'ORG', 'MONEY', 'PERSON', 'DATE', 'CARDINAL', 'TIME', 'NORP', 'ORDINAL'] 
        self.listNERCoeff   = 3 * np.ones(len(self.listNER)) #if each tag wants its own coefficient, hard-code the coefficient vector
        #number of speakers 
        self.numSpeakers    = 4           #set to -1 if to be determined from the transcript (not impl. yet) --> AMI Corpus has 4 speakers for all meetings
        #number of underlying topics 
        self.numTopics   = 16
        #high surprise value
        self.high           = 0
        self.small          = 0.00000000000001
        #segmentation
        self.numSegm        = None
        self.windowLength   = 5
        self.smoothParam    = 2
        self.kPeak          = 3
        #extract keywords parameters
        self.kMF            = 3
        self.kME            = 1
        self.Tm             = 0.5
        self.keywCoeff      = 1
        #monologue parameters
        self.monologueRatio = 0.5
        #dialogue parameters
        self.dialogueRatio  = 0.5
        self.wLex           = 0.5
        self.wTop           = 0.5
        self.alpha          = 0.9

        
    def ExtendStopwords(self):
        #add meeting domain specific stopowords
        [self.stopwords.append(s) for s in self.newStopwords]

    def CheckTopicModelSet(self):
        #check that only one topic model training set flag is set to true - if not, reset optimal config with topicModelSet2 = True
        if (self.topicModelSet1 + self.topicModelSet2 + self.topicModelSet3 != 1):
            self.topicModelSet1 = False
            self.topicModelSet2 = True
            self.topicModelSet3 = False


        

        
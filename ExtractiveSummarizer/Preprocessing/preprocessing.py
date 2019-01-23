import numpy as np
from config import Config

class Preprocessing(object):
    """It performs all the preprocessing steps required for meeting summarization"""
    def __init__(self):
        #config class
        cfg = Config()
        #class variables
        self.speakers  = []
        self.sentences = []
        self.sentLemma = []
        self.lemmaTags = []
        self.wordLemma = []
        self.wSpeakers = []
        self.nerWords  = []
        self.nerEnts   = []
        self.nlp       = cfg.nlp
        self.stopwords = cfg.stopwords
        self.puncList  = cfg.puncList
        self.pronLemma = cfg.pronLemma
        self.meetingHisto   = []
        self.singleWords    = []
        self.numSpeakers    = cfg.numSpeakers

    def Preprocess(self, transcript, c = 0):
        mergedSentences = ' '.join(transcript[1]) 
        oldSentences    = transcript[1]
        oldSpeakers     = transcript[0]

        fullText = self.nlp(mergedSentences)
        for ent in fullText.ents:
            self.nerWords.append(ent.text.lower())
            self.nerEnts.append(ent.label_)

        print("Named entities tagged ...")
        for sent in oldSentences:
            sent = self.nlp(sent)
            for temp in sent.sents:
                
                loc = []
                locTag = []
                for token in self.nlp(temp.text):
                    if self.Filter(token):
                        locTag.append([token.lemma_, token.tag_, token.dep_])
                        loc.append(token.lemma_)
                        self.wSpeakers.append(oldSpeakers[c])
                        self.wordLemma.append(token.lemma_) 
                if loc:
#                if len(loc) > 1: #if you want to filter out single non-stopword-based sentences
                    self.lemmaTags.append(locTag)
                    self.sentLemma.append(loc)
                    self.sentences.append(temp.text)
                    self.speakers.append(oldSpeakers[c])
                    
            c += 1
        print("Stopwords removed, lemmatization and POS tag completed ...")
        self.CreateMeetingHistogram()    
        print("Meeting histogram computed ...")
        
    
    def CreateMeetingHistogram(self):
        #computes the local histogram per each speaker
        #self.numSpeakers = int(np.max(self.speakers))
        for w in self.wordLemma:
           if w not in self.singleWords:
               self.singleWords.append(w) 
        self.meetingHisto = np.zeros((self.numSpeakers+1, len(self.singleWords)))
        c = 0
        for w in self.wordLemma:  
            spIdx = self.wSpeakers[c]
            idx   = self.singleWords.index(w)
            self.meetingHisto[0][idx]     += 1
            self.meetingHisto[spIdx][idx] += 1
            c += 1
    

    def TokenizeReference(self, ref):
        text = []
        ref = ''.join(ref)
        doc = self.nlp(ref1)
        for temp in doc.sents:
            lemma = []
            for token in self.nlp(temp.text):
                if self.Filter(token):
                    lemma.append(token.lemma_)
            if len(lemma) > 1:
                text.append(temp.text)
    
        return text

    def Filter(self, token):
        #filter stopwords, 
        return (token.text.lower() not in self.stopwords and token.text not in self.puncList and len(token.text) > 1 and token.lemma_ != self.pronLemma)


from config import Config
import numpy as np
from PaperTest.help import Help

class FrequencyMeasures(object):
    """Computes idf, tfidf, suidf"""
    def __init__(self, meetingHisto, meetingWords, wordsVector, Ns):
        #config class
        cfg = Config()
        #class variables
        self.meetingHisto = meetingHisto
        self.meetingWords = meetingWords
        self.wordsVector  = wordsVector
        self.Ns           = Ns
        self.high         = cfg.high
        self.small        = cfg.small
        self.tfidf        = []
        self.tfidfSpeak   = []
        self.idf          = []
        self.suidf        = []

    def GetAll(self):
        self.tfidf      = self.TfIdfGlobal(True)
        self.idf        = self.TfIdfGlobal(False)
        self.tfidfSpeak = self.TfIdfSpeakersMeeting()
        self.suidf      = self.Suidf()

    def TfIdfGlobal(self, tfTrue=True):
        tfidf = np.zeros(len(self.meetingWords))
        lenDataset = len(self.wordsVector)
        for x in range(len(self.meetingWords)):
            if tfTrue:
                tf = self.meetingHisto[0][x]/np.sum(self.meetingHisto[0][:])
            else:
                tf = 1
            count = 0
            for words in self.wordsVector:
                if self.meetingWords[x] in words:
                    count += 1
            try:    
                idf = np.log(len_dataset/count)     
            except:
                idf = 0
            
            tfidf[x] = tf * (1 + idf)
        return tfidf

    def TfIdfSpeakersMeeting(self): #all speakers

        tfidf = np.zeros((self.Ns, len(self.meetingWords)))

        for j in range(self.Ns):
            den = np.sum(self.meetingHisto[j+1][:])
            for x in range(len(self.meetingHisto[0])):
            
                tf = self.meetingHisto[j+1][x] / den #frequency from jth speaker
                count = 0
                for s in range(self.Ns):
                    if self.meetingHisto[s+1][x]:
                        count += 1
                try:
                    idf = np.log(self.Ns/count)
                except:
                    idf = self.small
                    tf = 1
                
                tfidf[j][x] = tf * (1 + idf)

        return tfidf

    #can receive idfVec
    def Suidf(self, idfVec=[]):      #computes suidf for all the words in the meeting (not in the dataset, not sure what's better)
        if ((not idfVec) and (not self.idf)):
            self.idf = self.TfIdfGlobal(False)
        
        
        surp_w_s = np.zeros((self.Ns, len(self.meetingWords))) #matrix [num_speakers X num_words]
        surp_w   = np.zeros(len(self.meetingWords))
        suidf_v  = surp_w
    
        for c in range(0, len(self.meetingWords)): #ext loop, it over words to match
        
            w_ref = self.meetingWords[c]
        
            for j in range(0, self.Ns):
                num = 0
                den = 0
                for k in range(0, self.Ns):
                    if j != k: 
                        num += self.meetingHisto[j+1][c]   #number of times speaker k+1 utters w_ref 
                        den += np.sum(self.meetingHisto[j+1][:])                #number of words uttered by given speaker, pass given sp and list of speak
                surp_w_s[j][c] = -np.log(Help.SafeDiv(num, den))
                if surp_w_s[j][c] == np.inf:
                    surp_w_s[j][c] = self.high * self.Ns
            
            
        for f in range(0, len(self.meetingWords)): #f idx of each single word
            word = self.meetingWords[f]
            summ = 0
            for c in range(0, self.Ns):
                summ += surp_w_s[c][f] 
            surp_w[f] = Help.SafeDiv(summ, self.Ns)
    #            suidf_v[f] = surp_w[f] * howmany(word, f, num_speak) * np.sqrt(idf(word)) / num_speak #howmany number of speaks uttered word
            suidf_v[f] = surp_w[f] * self.HowMany(f) * np.sqrt(self.idf[self.meetingWords.index(word)]) / self.Ns #howmany number of speaks uttered word    
        return suidf_v

    def HowMany(self, f, count = 0):
        for x in range(1, len(self.meetingHisto)):
           if self.meetingHisto[x][f]:
               count += 1
        return count


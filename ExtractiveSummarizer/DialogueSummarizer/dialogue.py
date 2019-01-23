import numpy as np
import math
from config import Config
from PaperTest.help import Help
from DialogueSummarizer.graphhelp import GraphHelp
from scipy import spatial as sp



class Dialogue(object):
    """Summarizes dialogues"""
#clean_sent[x], clean_speak[x], clean_sent_orig[x], sp_distr[x], top_term, top_doc, dictionary, words_vec, 
#speak_vec, num_topics, vocab, spacy_loc_single_w, corpus, spacy_list_words_vec, spacy_list_histo_vec, 
#topic_keyw_idx, topic_keyw_weights, ner_w, ner_ent, list_NER, list_NER_coeff) #corpus is BoW

    def __init__(self, prep, segm, histo, topic, keyw, freq, speakFreq):
        #config class
        cfg             = Config()
        #class variables
        self.ratio      = cfg.dialogueRatio
        self.listNer    = cfg.listNER
        self.nerCoeff   = cfg.listNERCoeff
        self.wLex       = cfg.wLex
        self.wTop       = cfg.wTop
        self.small      = cfg.small
        self.alpha      = cfg.alpha
        self.prep       = prep
        self.segm       = segm
        self.topicModel = topic
        self.histograms = histo
        self.keywords   = keyw
        self.freqVec    = freq
        self.speakVec   = speakFreq
        #results
        self.summary    = []

    def Summarize(self):

        [Luu, Lss, Lus] = self.CreateMatricesLayers()
        score = self.TwoLayer(Luu, Lss, Lus)
        self.summary   = self.ExtractSummary(score)

    def CreateMatricesLayers(self):
        #build edge weights
        LuuLex = self.CreateLuu(top=False, lex=True)
        LuuTop = self.CreateLuu(top=True, lex=False)
        Luu    = self.wLex * LuuLex + self.wTop * LuuTop
        Lss    = self.CreateLss()
        Lus    = self.CreateLus()
        #remove empty cols
        Lss    = GraphHelp.RemoveEmptyCols(Lss)
        Lus    = GraphHelp.RemoveEmptyCols(Lus, squared=False)
    
        return [Luu, Lss, Lus]    


    def CreateLuu(self, top=False, lex=True): # top=True means the function computes topical similarity, else lex=True computes lexical similarity                                                                   
         
        Luu = np.zeros((len(self.segm.cleanSentences),len(self.segm.cleanSentences))) # matrix [num_utterances X num_utterances]
        if (top and lex) or ((not top) and (not lex)):  # if error in passing parameters (Luu can be based only on one kind of similarity)
            top = False                                 # reset default parameters
            lex = True                                  # reset default parameters
    
        if top: #topic similarity
        
            prob_top_sent = np.zeros((len(self.topicModel['Terms']), len(self.segm.cleanSentences)))
            for x in range(len(self.topicModel['Terms'])):
                for y in range(len(self.segm.cleanSentences)):
                    num = 0
                    den = 0
                    for w in self.segm.cleanSentences[y]:
    #                    idx_w = find_index_word(w, corpus, tokens_topic_model) #ret -1 if w not in corpus
                    
                        try:
                            tk_id = self.topicModel['Dictionary'].token2id[w]
                            num += (Help.FreqWordInSentence(w, self.segm.cleanSentences[y]) * self.topicModel['Terms'][x][tk_id])
                        except:
                            num += (Help.FreqWordInSentence(w, self.segm.cleanSentences[y]) * self.small)
                        den += Help.FreqWordInSentence(w, self.segm.cleanSentences[y])
                    prob_top_sent[x][y] = Help.SafeDiv(num, den)
        
            for x in range(len(self.segm.cleanSentences)):
                for y in range(len(self.segm.cleanSentences)):
                    LTS_sum = 0
                    prob = 0
                    for w in self.segm.cleanSentences[y]:
           
                        wFreq = self.ComputeTermFrequency(w) #creates a vector with the frequency of the word per each doc
                        if np.sum(w_freq): #if w doesn't appear in the dictionary, don't waste time
                            LTS_sum += self.CopmputeLTS(wFreq)  #return sum over all topics (LTS of a single word with frequency term_freq)
                    prob = Help.SumTopics(prob_top_sent, x) #should I pass x or y?
                    Luu[x][y] = LTS_sum * prob
    
        else: #lexical similarity
            for i in range(len(self.segm.cleanSentences)):
                v1 = Help.CreateSentenceVector(self.segm.cleanSentences[i], self.freqVec, self.prep.singleWords) 
                for j in range(0, len(self.segm.cleanSentences)):
                    v2 = Help.CreateSentenceVector(self.segm.cleanSentences[j], self.freqVec, self.prep.singleWords)
                    if Help.NotValidCos(v1, v2):
                        v1, v2 = Help.ReshapeVec(v1, v2)
                        #if complains about v1, v2 dimensions, add zeros (or ones, idk yet, is a cosine distance) to match those dimensions 
                    cos_sim = 1 - sp.distance.cosine(v1, v2)
                    if math.isnan(cos_sim):
                        Luu[i][j] = 0.
                    else:
                        Luu[i][j] = cos_sim 
    
                    # cosine similarity only if vectors of same size 
    #    return norm(Luu, norm='l1') #matrix representing lexical (topic) similarity via word overlap (via LDA)
        return Luu 

    def CopmputeLTS(self, wFreq):
        #topicModel['Docs'] is a list of len(num_documents) where each el is (topic_id, topic_prob)
                                                    # wFreq is a list with size #documents giving the frequency of the term in each doc
        LTS = np.zeros(self.topicModel['NumTopics'], dtype=float)
        num = np.zeros(self.topicModel['NumTopics'], dtype=float)
        den = np.zeros(self.topicModel['NumTopics'], dtype=float)
    
        for y in range(len(self.topicModel['Docs'])): #iterate over documents
            if wFreq[y]:
                for x in range(len(self.topicModel['Docs'][y])):  
        
                    idx = self.topicModel['Docs'][y][x][0] -1
                
                    prob = self.topicModel['Docs'][y][x][1]
                    add_num = wFreq[y] * prob
                
                    add_den = wFreq[y] * (1 - prob)
                
                    num[idx] = num[idx] + add_num
                    den[idx] = den[idx] + add_den
                
        for x in range(len(LTS)):
            LTS[x] = Help.SafeDiv(num[x], den[x]) #first sum over docs, then divide, then sum over topics

        return np.sum(LTS) #return sum over all topics (LTS of a single word with frequency term_freq)


    def ComputeTermFrequency(self, w, flag=True):
        freq = []
        try:
            w_id = self.topicModel['Vocab'][w]
        except:
            return 0
    
        if flag:
            for doc in self.topicModel['Corpus']:
                flag = False
                for el in doc:
                    if el[0] == w_id:
                        freq.append(el[1])
                        flag = True
                        break
                if not flag:
                    freq.append(0)
    
        return freq
    
    def CreateLss(self): #freq vec is the speaker tfidf
        Ns = self.prep.numSpeakers
        Lss = np.zeros((Ns, Ns))

        for i in range(0, Ns):
            if i+1 in self.segm.cleanSpeakers:
                v1 = Help.CreateSpeakerVector(i, self.segm.cleanSentences, self.segm.cleanSpeakers, self.speakVec)
                for j in range(0, Ns):
                    if j+1 in self.segm.cleanSpeakers:                
                        v2 = Help.CreateSpeakerVector(j, self.segm.cleanSentences, self.segm.cleanSpeakers, self.speakVec)
                        if Help.NotValidCos(v1, v2):
                            v1, v2 = Help.ReshapeVec(v1, v2)  
                        cos_dist = 1 - sp.distance.cosine(v1, v2)
                        if math.isnan(cos_dist):
                            Lss[i][j] = 0.
                        else:
                            Lss[i][j] = cos_dist 
    
        return Lss 

    def create_Lus(self):
        Ns = self.prep.numSpeakers
        Lus = np.zeros((len(self.segm.cleanSentences), Ns))
        for i in range(0, len(self.segm.cleanSentences)):
            v1 = Help.CreateSentenceVector(self.segm.cleanSentences[i], self.freqVec, self.prep.singleWords)
            for j in range(0, Ns):
                if j+1 in self.segm.cleanSpeakers:
                    v2 = Help.CreateSpeakerVector(j, self.segm.cleanSentences, self.segm.cleanSpeakers, self.speakVec)
                    if Help.NotValidCos(v1, v2):
                        v1, v2 = ReshapeVec(v1, v2)  
                    cos_dist = 1 - sp.distance.cosine(v1, v2)
                    if math.isnan(cos_dist):
                        Lus[i][j] = 0.
                    else:
                        Lus[i][j] = cos_dist 
        return Lus 

      
    def TwoLayer(self, Luu, Lss, Lus):
        #checks and preprocessing
        [L_11, L_22, L_12, L_21, num1, num2] = GraphHelp.Preprocess(Luu, Lss, Lus)
    
        #intialization
        S1 = np.ones(num1)/num1
        for x in range(num1):
            accum_topic = 0
            for w in sentences[x]:
                if w in self.prep.nerWords:
                    idx = self.prep.nerWords.index(w)
                    word_ent = self.prep.nerEnts[idx]
                    try:
                        S1[x] = S1[x] * self.nerCoeff[self.listNer.index(word_ent)]
                    except:
                        ...
            
        S1 = S1 / np.sum(S1) 
        if np.sum(S1) < 0.99 or np.sum(S1) > 1.01:
            print('NORMALIZATION NOT CORRECT!\n')
            print(np.sum(S1))

        S1 = S1[:, np.newaxis]
        
        S2 = np.ones(num2)/num2
        S2 = S2[:, np.newaxis]
    
        former = (1 - self.alpha) * S1 + self.alpha * (1 - self.alpha) * np.dot(L_11, np.dot(L_12, S2))
        latter = self.alpha * self.alpha * np.dot(L_11, np.dot(L_12, np.dot(L_22, L_21)))
        combine = former * np.ones(num1) + latter

        try:
            w, v = np.linalg.eig(combine)
            score = abs(v[:, 0])/sum(abs(v[:, 0]))
        
            return score/np.sum(score)
        except:
            return S1





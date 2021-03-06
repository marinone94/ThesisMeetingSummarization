import numpy as np
import math
from config import Config
from PaperTest.help import Help
from DialogueSummarizer.graphhelp import GraphHelp
from scipy import spatial as sp



class Dialogue(object):
    """Summarizes dialogues"""


    def __init__(self, prep, segm, histo, topic, freq, speakFreq, iter):
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
        self.freqVec    = freq
        self.speakVec   = speakFreq
        self.íter       = iter
        #results
        self.summary    = []

    def Summarize(self):

        [Luu, Lss, Lus] = self.CreateMatricesLayers()
        score = self.TwoLayer(Luu, Lss, Lus)
        self.summary   = self.ExtractSummary(score)

    def ExtractSummary(self, score_utt):
        temp_dialogue = []
        dialogue = []
        num_words = 0
        for s in self.segm.cleanSentences[self.íter]:
            num_words = num_words + len(s)
    
        len_summ = int (self.ratio * num_words)
        sent_idx = np.argsort(score_utt)
        rev_idx = sent_idx[::-1]
    
        words_in_summ = 0
        for i in rev_idx:
            temp_sent = self.segm.cleanSentOrig[self.íter][i]
            if temp_sent not in temp_dialogue:
                temp_dialogue.append(temp_sent)
            words_in_summ = words_in_summ + len(self.segm.cleanSentences[self.íter][i])
            if words_in_summ > len_summ: #append also the first sentence which exceeds the summary length
                break
    
        for sent in self.segm.cleanSentOrig[self.íter]:
            if sent in temp_dialogue:
                dialogue.append(sent)
            else:
                dialogue.append(' ')
       
        dialogue.append("\n")
        return ' '.join(dialogue)


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
         
        Luu = np.zeros((len(self.segm.cleanSentences[self.íter]),len(self.segm.cleanSentences[self.íter]))) # matrix [num_utterances X num_utterances]
        if (top and lex) or ((not top) and (not lex)):  # if error in passing parameters (Luu can be based only on one kind of similarity)
            top = False                                 # reset default parameters
            lex = True                                  # reset default parameters
    
        if top: #topic similarity
        
            prob_top_sent = np.zeros((len(self.topicModel['Terms']), len(self.segm.cleanSentences[self.íter])))
            for x in range(len(self.topicModel['Terms'])):
                for y in range(len(self.segm.cleanSentences[self.íter])):
                    num = 0
                    den = 0
                    for w in self.segm.cleanSentences[self.íter][y]:
    #                    idx_w = find_index_word(w, corpus, tokens_topic_model) #ret -1 if w not in corpus
                    
                        try:
                            tk_id = self.topicModel['Dictionary'].token2id[w]
                            num += (Help.FreqWordInSentence(w, self.segm.cleanSentences[self.íter][y]) * self.topicModel['Terms'][x][tk_id])
                        except:
                            num += (Help.FreqWordInSentence(w, self.segm.cleanSentences[self.íter][y]) * self.small)
                        den += Help.FreqWordInSentence(w, self.segm.cleanSentences[self.íter][y])
                    prob_top_sent[x][y] = Help.SafeDiv(num, den)
        
            for x in range(len(self.segm.cleanSentences[self.íter])):
                for y in range(len(self.segm.cleanSentences[self.íter])):
                    LTS_sum = 0
                    prob = 0
                    for w in self.segm.cleanSentences[self.íter][y]:
           
                        wFreq = self.ComputeTermFrequency(w) #creates a vector with the frequency of the word per each doc
                        if np.sum(wFreq): #if w doesn't appear in the dictionary, don't waste time
                            LTS_sum += self.CopmputeLTS(wFreq)  #return sum over all topics (LTS of a single word with frequency term_freq)
                    prob = Help.SumTopics(prob_top_sent, x) #should I pass x or y?
                    Luu[x][y] = LTS_sum * prob
    
        else: #lexical similarity
            for i in range(len(self.segm.cleanSentences[self.íter])):
                v1 = Help.CreateSentenceVector(self.segm.cleanSentences[self.íter][i], self.freqVec, self.prep.singleWords) 
                for j in range(len(self.segm.cleanSentences[self.íter])):
                    v2 = Help.CreateSentenceVector(self.segm.cleanSentences[self.íter][j], self.freqVec, self.prep.singleWords)
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
            if i+1 in self.segm.cleanSpeakers[self.íter]:
                v1 = Help.CreateSpeakerVector(i, self.segm.cleanSentences[self.íter], self.segm.cleanSpeakers[self.íter], self.speakVec)
                for j in range(0, Ns):
                    if j+1 in self.segm.cleanSpeakers[self.íter]:                
                        v2 = Help.CreateSpeakerVector(j, self.segm.cleanSentences[self.íter], self.segm.cleanSpeakers[self.íter], self.speakVec)
                        if Help.NotValidCos(v1, v2):
                            v1, v2 = Help.ReshapeVec(v1, v2)  
                        cos_dist = 1 - sp.distance.cosine(v1, v2)
                        if math.isnan(cos_dist):
                            Lss[i][j] = 0.
                        else:
                            Lss[i][j] = cos_dist 
    
        return Lss 

    def CreateLus(self):
        Ns = self.prep.numSpeakers
        Lus = np.zeros((len(self.segm.cleanSentences[self.íter]), Ns))
        for i in range(0, len(self.segm.cleanSentences[self.íter])):
            v1 = Help.CreateSentenceVector(self.segm.cleanSentences[self.íter][i], self.freqVec, self.prep.singleWords)
            for j in range(0, Ns):
                if j+1 in self.segm.cleanSpeakers[self.íter]:
                    v2 = Help.CreateSpeakerVector(j, self.segm.cleanSentences[self.íter], self.segm.cleanSpeakers[self.íter], self.speakVec)
                    if Help.NotValidCos(v1, v2):
                        v1, v2 = Help.ReshapeVec(v1, v2)  
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
            for w in self.segm.cleanSentences[self.íter][x]:
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





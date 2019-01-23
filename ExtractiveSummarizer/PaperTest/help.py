import numpy as np
import itertools
import math
from config import Config

class Help(object):
    """Support class"""

    def GenCat(Ns):
        return list(itertools.product([0, 1], repeat=Ns))
    
    def SafeDiv(x, y):
        if not y:
            return 0
        return x/y

    def CreateSentenceVector(sent, freqVec, singleWords):
        
        vec = np.zeros(len(sent))
        for x in range(len(sent)):
            s = sent[x]
            summ = 0
            for w in s:
                summ = freqVec[singleWords.index(w)]
            vec[x] = summ
        return vec

    def CreateSpeakerVector(j, sent, speaks, freq_vec, idf_vec = []):
        cfg = Config()
        
        for y in range(len(sent)):
            sp = speaks[y]
            ss = sent[y]
            for x in range(len(ss)):
                w = ss[x]
                if sp == j+1:
                    try:
                        idf_vec.append(freq_vec[j][spacy_loc_single_w.index(w)])
                    except:
                        idf_vec.append(cfg.small) #or should it be 1?
                    
        return idf_vec
    
    #jeffrey divergence
    def Dist(p, q):

        dist = 0
        length = len(p) #vector size
        for i in range(0, length):
            P = p[i]
            Q = q[i]
            summ = P + Q    
            log_p = np.log(Help.SafeDiv(P * 2, summ))
            log_q = np.log(Help.SafeDiv(Q * 2, summ))
            if log_p != -math.inf: 
                temp_p = P * log_p
            else:
                temp_p = 0
            if log_q != -math.inf:            
                temp_q =  Q * log_q
            else:
                temp_q = 0

            dist += (temp_p + temp_q)

        return dist

    def Dstr(text, num_speak):
        tot_w = 0
        dstr_vec = np.zeros(num_speak)
        for j in range(0, num_speak):
            par = Help.WordsInSegm(text[0], text[1], j)
            tot_w += par
            dstr_vec[j] = par # num of words uttered by s in the sgem        
        try:
            return dstr_vec/tot_w
        except:
            return dstr_vec/1

    def WordsInSegm(sentences, speakers, s): #words uttered by s in a list of sentences (words_s() doesn't fit this case)
        num = 0
        for c in range(0, len(sentences)):
            if speakers[c] == s+1:
                num += len(sentences[c])
        return num

    def DstrId(Ns, speakers): #speakers array of booleans whether the speaker is active or not in this segment 
                          
        dstr_id = np.zeros(Ns)
        act_speakers = np.sum(speakers) #num of active speakers in the segment
        for x in range(0, Ns):
            if speakers[x]: #uniformly distributed over active speakers
                dstr_id[x] = 1/act_speakers
                                
        return dstr_id 

    def RemoveMinorSpeaker(text, sp, dstr, clean = []):
         #if at position sp[x]-1 dstr is true append text (lemmatized sentence, speaker or original sentence)
        [clean.append(text[x]) for x in range(len(sp)) if dstr[sp[x]-1]]
        return clean    

    def GenSpeaksWordsTag(segments, speakers, tag, segm_tag = [], segm_sp = [], segm_w = []): #from a window of sentences, generate two vectors speaks and words    
 
        for d in range(0, len(segments)):
            w = segments[d]
            t = tag[d]
            s = speakers[d]
            words = []
            speaks = []
            tags = []
            for c in range(0, len(w)):    
                for k in range(len(w[c])):
                        words.append(w[c][k])
                        speaks.append(s[c])
                        tags.append(t[c][k])
            segm_w.append(words)
            segm_tag.append(tags)
            segm_sp.append(speaks)
        
        return [segm_tag, segm_sp, segm_w]
    

    def Expand(segm, word = [], s = [], t = []):
        
        for el in segm[0]:
            for w in el:
                t.append(w)
        for el in segm[1]:
           for w in el:
                s.append(w)
        for el in segm[2]:
           for w in el:
                word.append(w)
        return [word,s,t]

    def WordSegmFrequency(tag, ref): #tag list of [words, POS] of given segments
        count = 0
        for w in tag: #tag list of words of given segment
            if w.lower() == ref.lower():
                count += 1            
        return count

    def RemoveMultipleKeywords(arr, ext_score, words = [], score = []):
        for x in range(len(arr)):
            if arr[x] not in words:
                words.append(arr[x])
                score.append(ext_score[x])
        return [words, score]

    def ReshapeVec(x, y):
    
        lx = len(x)
        ly = len(y)
    
        count = 0
        if lx < ly:
            y = y[:lx]
        elif lx > ly:
            x = x[:ly]
        return [x, y]
    
    def SumTopics(prob_matr, y, res = 0):
    
        for x in range(len(prob_matr)):
            res += prob_matr[x][y]
        
        return res

    def NotValidCos(x,y):
        return (not(len(x) == len(y)))

    

    def FreqWordInSentence(w, sent, freq = 0):
        for word in sent:
            if w == word:
                freq += 1
        return freq
    

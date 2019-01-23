from config import Config
from PaperTest.help import Help
import numpy as np


class FuncSegm(object):
    """ This class divides a transcript into monologues and dialogues
        Identifies the active speakers
        Filter out sentences uttered by non-active ones
    """
    def __init__(self, prep, freq, Ns): #freq is suidf vector
        #config class
        cfg = Config()
        #class variables
        self.cleanSentences = []
        self.cleanSpeakers  = []
        self.cleanSentOrig  = []
        self.cleanSentTags  = []
        self.speakerDistr   = []
        self.candidateBound = []
        self.boundaries     = []
        
        #preprocessed variables
        self.prep           = prep
        #frequency vectors
        self.freq           = freq
        self.Ns             = Ns
        #segmentation parameters
        self.numSegm        = cfg.numSegm
        self.winLength      = cfg.windowLength
        self.smoothParam    = cfg.smoothParam
        self.kPeak          = cfg.kPeak
    
    def Segmentation(self):
        #get boundaries, CB and number of segments
        self.GetBoundaries()#
        #segment according to previous results
        newSegments         = self.Segmenter()#

        for x in range(len(newSegments[0])):
            locSentences = newSegments[0][x]
            locSpeakers = newSegments[1][x]  
            vec = [sent, sp]
            self.speakerDistr.append(self.ScoreSegment(vec)[1])#

        for x in range(len(newSegments[0])):
            self.cleanSentences.append(Help.RemoveMinorSpeaker(newSegments[0][x], newSegments[1][x], self.speakerDistr[x]))    
            self.cleanSpeakers.append(Help.RemoveMinorSpeaker(newSegments[1][x], newSegments[1][x], self.speakerDistr[x]))  
            self.cleanSentOrig.append(Help.RemoveMinorSpeaker(newSegments[2][x], newSegments[1][x], self.speakerDistr[x]))  
            self.cleanSentTags.append(Help.RemoveMinorSpeaker(newSegments[3][x], newSegments[1][x], self.speakerDistr[x]))  

        
    def ScoreSegment(self, segm, sum_c = [], c_idx = []):
        cat = Help.GenCat(self.Ns) #generate categories vector
        cat = cat[1:] #[0,0,0,0,0] not allowed, there's always at least one speaker
 
        #sum_c score per each cat, find min
        #c_idx idx of each cat, to find the cat corresp to min score
        for c in cat:

            sum_dist = 0   
            loc_segm = [segm[0][i:i+M], segm[1][i:i+M]]
            sum_dist = Dist(Dstr(loc_segm, self.Ns), DstrId(self.Ns,c)) #score segments
            sum_c.append(sum_dist)
            c_idx.append(c)
             #c single category boolean vector used for calling dstr_id
        
        min_score = np.min(sum_c) 
        min_cat = c_idx[np.argmin(sum_c)] # in the list of categories, take el of idx that minimizes the score         

        return min_score, min_cat
    
    def Segmenter(self):
        test_transcript_0 = [self.prep.sentLemma, self.prep.speakers, self.prep.sentences, self.prep.lemmaTags]
        text_test = []
        speak_test = []
        text_orig_test = []
        segm_idx = []
        tag_test = []
        for el in self.boundaries:
            segm_idx.append(int(el))
        for c in range(0, len(segm_idx)):
            if c == 0:
                text_test.append(test_transcript_0[0][:segm_idx[c]])
                speak_test.append(test_transcript_0[1][:segm_idx[c]])
                text_orig_test.append(test_transcript_0[2][:segm_idx[c]])
                tag_test.append(test_transcript_0[3][:segm_idx[c]])
            else:
                text_test.append(test_transcript_0[0][segm_idx[c-1]:segm_idx[c]])
                speak_test.append(test_transcript_0[1][segm_idx[c-1]:segm_idx[c]])
                text_orig_test.append(test_transcript_0[2][segm_idx[c-1]:segm_idx[c]])
                tag_test.append(test_transcript_0[3][segm_idx[c-1]:segm_idx[c]])
        test_segm = []
        test_segm.append(text_test)
        test_segm.append(speak_test)
        test_segm.append(text_orig_test)
        test_segm.append(tag_test)
        return test_segm
    
        
    def GetBoundaries(self, peaks = []):
        #get boundaries, CB and number of segments
        lenText = len(self.sentences) 
    
        [scores, smoothedScores] = self.Score(lenText)
      
        [peaks.append(self.FindPeak(scores, x)) for x in range(len(scores))]
        [self.candidateBound.append(x) for x in range(len(peaks)) if peaks[x]] 

        peakScores = self.PeakScore(lenText, smoothedScores)
        loc_segm = self.Segm(num_segm, peakScores)
    
   

    def FindPeak(self, scores, x):
        return ((score[x] > score[x-1]) and (score[x] > score[x+1]))

    def FindValley(self, score, x, direct):
        if direct == '+':
            score = score[x:]
        elif direct == '-':
            score = score[::-1]
            score = score[-x-1:]
        else:
            raise ImplementationError('Direction not assigned!') #DIRECTION NOT ASSIGNED!!!
        
        for y in range(1, len(score)-1): #first and last element are not considered as possible valleys
            pr = False
            af = False
            if score[y] < score[y-1]:
                pr = True
            if score[y] < score[y+1]:
                af = True
            if pr and af:
                return y #distance of valley from given peak
    
        return len(score) -1  #if doeesn't find any valley, return the last index cause it means it's monotonically decreasing

    def Segm(self, peakScoreIdx): 
    
        peak_score = peakScoreIdx[0]
        idx        = peakScoreIdx[1] #reference to original index of segment
      
        if self.numSegm == None:
            av_peak = np.average(peak_score) #avg peak scores
            stddev_peak = np.std(peak_score) #std dev peak scores
            thresh = self.kPeak * (av_peak - stddev_peak) #threshold for finding boundaries 
            #return only top scored peaks depending on threshold if num of segments unknown
            [self.boundaries.append(idx[x]) for x in range(0, len(peak_score)) if peak_score[x] > thresh]
            self.numSegm = len(self.boundaries)
        else:
            sorted_peaks = peak_score.sort() # evaluate if peaks may get the same score, don't think so actually
            top_peaks = sorted_peaks[:num_segm] #take top num_segm peak scores from sorted score peaks vector
        
            [self.boundaries.append(idx[x]) for x in range(len(peak_score)) if peak_score[x] in top_peaks]
    

    def PeakScore(self, lenText, score, j = 0): 
    
        peaks = np.zeros(lenText)
        nw    = np.zeros(lenText)
        pw    = np.zeros(lenText)
         #less peaks than possible boundaries
    
        for x in range(len(score)):
            peaks[x] = self.FindPeak(score, x)
            if peaks[x]: #find valleys just when there's a peak
                pw[x] = self.FindValley(score, x, '-')
                nw[x] = self.FindValley(score, x, '+') #plus means next valley
            else: 
                pw[x] = -1
                nw[x] = -1
    
        peakScore = np.zeros((2, int(np.sum(peaks)))) #peaks is binary vector, peak score has size equal to nonzero el in peaks
                                                       # 2nd row for corresponding idx in peaks (same as list of sentences, each may be a boundary)
        for i in range(len(peaks)):
            if peaks[i]: #out of peaks, score is zero 
                a = score[i]
                b = score[int(i - pw[i])]
                c = score[int(i + nw[i])]
                peakScore[0][j] = 2 * a - b - c #vector of n_peaks elements containing score for each
                peakScore[1][j] = i 
                j += 1 #j to move on indeces of peaks, i for scores

        return peakScore #matrix of [peak scores; idxs]

    def Score(self, lenText):    

        score = np.zeros(lenText)
        smooth_score = np.zeros(lenText)
        for i in range(1, lenText-1): # iterate over all wiindows (as many as num of sentences), 
                                       # each can be potential segment boundary but first and last sentences             
            maxim_i = np.maximum(0, i-self.winLength)
            maxim_f = np.maximum(1, i)
            minim = np.maximum(np.minimum(lenText, i + self.winLength), maxim_f)  
            win_left_idx = []
            win_right_idx = []
        
            for x in range(maxim_i, maxim_f): # min and max to avoid index problems
                win_left_idx.append(x)
            
            for x in range(i, minim): # min and max to avoid index problems
                win_right_idx.append(x)
        
            if i > 1:
                win_l = [self.prep.speakers[win_left_idx[0] : win_left_idx[-1]], self.prep.sentences[win_left_idx[0] : win_left_idx[-1]]] #create win left_i
            else: #when i = 1 --> win_l is only the first el 
                win_l = [[self.prep.speakers[0]], [self.prep.sentences[0]]]
            if i < lenText - 2:
                win_r = [self.prep.speakers[win_right_idx[0] : win_right_idx[-1] +1], self.prep.sentences[win_right_idx[0] : win_right_idx[-1] +1]] #create win left_i
            else:
                win_r = [[self.prep.speakers[-1]], [self.prep.sentences[-1]]]
        
            win_words_l = self.GenSpeaksWords(win_l) #win_words = [speaks, words] of a given window
            s_l = win_words_l[0] #list of speakers
            w_l = win_words_l[1] #list of words
            wnd_l = self.RemoveDupl(w_l) #remove duplicates and punctuation 

            win_words_r = self.GenSpeaksWords(win_r) #win_words = [speaks, words] of a given window
            s_r = win_words_r[0] #list of speakers
            w_r = win_words_r[1] #list of words
            wnd_r = self.RemoveDupl(w_r) #remove duplicates and punctuation, lose order 

            WC_l = self.WC(win_l)
            WC_r = self.WC(win_r)
            dist_wc = dist(WC_l, WC_r)
            WI_l = self.WI(w_l, s_l, wnd_l)
            WI_r = self.WI(w_r, s_r, wnd_r)
            dist_wi = dist(WI_l, WI_r)
            score[i] = dist_wc + dist_wi  
    
        for i in range(1, lenText-1):
            temp_score = 0 #loc score for smoothing
            bound = self.SafeSmooth(i, len(score)) #check not to go out of size when smoothing
            low = bound[0]
            up = bound[1]
            for j in range(low, up): 
                temp_score += score[j] 
            smooth_score[i] = temp_score / (1 + self.smoothParam)
        return score, smooth_score

    def WI(w, s, wnd):    
        
        WI_vec = np.zeros(self.Ns)
        if w:
            suidf_win = CreateSentenceVector(w, self.freq, self.prep.singleWords)
        
            den = 0 #doesn't have to be reset
            num = [] #append num per each speaker
            for j in range(0, self.Ns):
                num_t = 0 #numerator for given speaker
                for k in range(0, len(s)):
                    idx = wnd.index(w[k])
                    if idx >= 0:
                        if s[k] == j+1:
                            num_t += suidf_win[idx]
                            den += suidf_win[idx]   
                num.append(num_t)
            
        
            for j in range(0, self.Ns):
                WI_vec[j] = SafeDiv(num[j] , den)
        
        return WI_vec  
    
    def WC(self, win):
    
        WC_vec = np.zeros(self.Ns) #don't need a matrix, store the result in score vector
        length = 0

        for s in win[1]:
            length = length + len(s)    
    
        for j in range(0, self.Ns): #WC_j left in i 
            count = 0
            x = 0
            for s in win[0]:
                if s == j+1:
                    count = count + len(win[1][x]) # number of words uttered by j in win
                x += 1 
            WC_vec[j] = SafeDiv(count, length)    #match with dimensionality            
        return WC_vec    
    
    def RemoveDupl(self, words, new = []):
        [new.append(w) for w in words if w not in new]
        return new

    def GenSpeaksWords(self, win, words = [], speaks = [], c = 0): #from a window of sentences, generate two vectors speaks and words    
    
        s = win[0]
        w = win[1]
    
        if len(s) == 1:
            for x in range(len(w[0])):
                speaks.append(s[0])
            words = w[0]
        else:
            
            for sent in w:
            
                for el in sent:
                    words.append(el)
                    speaks.append(s[c])
                c += 1   
 
        return [speaks, words]        
    
    def SafeSmooth(self, i, length):
        low = int(i - self.smoothParam/2)
        if low < 0:
            low = 0
   
        up = int(i + self.smoothParam/2 +1)
        if up > length:
            up = length
    
        return [low, up]



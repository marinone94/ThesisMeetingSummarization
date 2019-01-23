class Extractor(object):
    """Extract keywords from meeting"""

    def __init__(self, prep, segm, freq):
        #config class
        cfg             = Config()
        #parameters
        self.listNer    = cfg.listNER
        self.nerCoeff   = cfg.listNERCoeff
        self.allowedPOS = cfg.allowedPOS
        self.kIdf       = cfg.kMF
        self.kNent      = cfg.kME
        self.Tm         = cfg.Tm
        self.keywCoeff  = cfg.keywCoeff
        #class variables
        self.keywords   = []
        self.keywScores = []
        #preprocessed variables
        self.prep       = prep
        #frequency vector
        self.freqVec    = freq
        #segmented
        self.segm       = segm

    def ExtractKeywords(self):
        tk_segm = GenSpeaksWordsTag(self.segm.cleanSentences, self.segm.cleanSpeakers, self.segm.cleanSentTags) #generate list of tokenized words with relative speaker UNCOMMENT
        tk_segm_list = Expand(tk_segm) #words, speakers, tags
        idf_w = np.zeros(len(tk_segm_list[0]))
        ext_score = np.zeros(len(tk_segm_list[0]))
        num_segm = len(segments)
        prob_s_w = np.zeros((len(tk_segm_list[0]), num_segm))
        num_s_w = np.zeros((len(tk_segm_list[0]), num_segm))
        nent  = np.zeros(len(tk_segm_list[0]))
        den_w = np.zeros(len(tk_segm_list[0]))

        #tfidf is created considering the words ordered as in spacy_loc_single_w
        [idf_w[i] = tfidf_vec[self.prep.singleWords.index(tk_segm_list[0][i])] for i in range(0, len(tk_segm_list[0]))]

        #check this later
        for i in range(0, len(tk_segm_list[0])): #i iter over words
            for j in range(0, num_segm): #j iter over segments
                try:
                    freq = WordSegmFrequency(tk_segm[2][j], tk_segm_list[0][i]) #times w is uttered in segment j
                except:
                    print(j)
                    input("enter")
                try:
                    num_s_w[i][j] = freq
                except:
                    print("2")
                    input("enter")
                try:
                    den_w[i] = freq + den_w[i] #sum over all segments
                except:
                    print("3")
                    input("enter")
                
                
        for i in range(0, len(tk_segm_list[0])):
            for j in range(0, num_segm):
                prob_s_w[i][j] = num_s_w[i][j] / den_w[i] #probability of being in segment j given word i
    
        for i in range(0, len(tk_segm_list[0])):
            for j in range(0, num_segm):
                if tk_segm_list[0][i] in tk_segm[2][j]:
                    nent[i] = nent[i] - (prob_s_w[i][j] * np.log(prob_s_w[i][j])) #negative entropy of each word
    
        [ext_score[i] = k_idf * idf_w[i] + k_nent * nent[i] for i in range(0, len(tk_segm_list[0]))[ #sum global and local scores
        [loc_single_words, scores] = RemoveMultipleKeywords(tk_segm_list[0], ext_score) #remove multiple keywords
        newScores = self.PosAndNer(loc_single_words, scores, tk_segm_list) #filter according to POS and adapt weight acc to NER
        self.Revert(loc_single_words, newScores) #revert scores and extract top Tm%
        
    
    def Revert(self, loc_single_words, newScores):
        #revert scores and extract top Tm%
        top_percent = int(self.Tm * len(loc_single_words))            
        sorted_scores_rev = sorted(newScores, reverse=True) #small to big
        self.keywScores = sorted_scores_rev[:top_percent] # sort scores bigger to smaller
        sorted_idx_rev = np.argsort(newScores)
        sorted_idx = sorted_idx_rev[::-1][:top_percent]
        for idx in sorted_idx:
            self.keywords.append(loc_single_words[idx])

    
    def PosAndNer(self, loc_single_words, scores, tk_segm_list)
        
        for y in range(0, len(loc_single_words)):
            for x in range(0, len(tk_segm_list[2])):
                if tk_segm_list[0][x].lower() == loc_single_words[y].lower():
    #                idx = find_idx_word(NER_words, tk_segm[x])  #if tk_segm[x] is in the list of specific NERs, extract its index 
                    word = tk_segm_list[0][x].lower()
                    idx_POS = tk_segm_list[2][x][1]
                    try:
                        ner_entity = self.prep.nerEnts[self.prep.nerWords.index(word.lower())] #ner_w list of words, ner_ent list of relative entities
    #                    print(" entity assigned to")
    #                    print(word)
                    except:
                        ner_entity = 'null'
                    if idx_POS in self.allowedPOS:
                        if ner_entity in self.listNer: #if label is in the list of good labels, increase the score
                        
                            scores[y] = scores[y] * self.nerCoeff[self.listNer.index(ner_entity)] #inc_score param to be tuned    
                        scores[y] = scores[y] * self.keywCoeff
                        self.keywCoeff = 1 #reset it for new words
                    break
                
        return scores



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
        self.numSpeakers    = []

    def Preprocess(self, transcript, c = 0):
        mergedSentences = ' '.join(transcript[1]) 
        oldSentences    = transcript[1]
        oldSpeakers     = transcript[0]

        fullText = self.nlp(mergedSentences)
        for ent in fullText.ents:
            self.nerWords.append(ent.text.lower())
            self.nerEnts.append(ent.label_)

 
        for sent in oldSentences:
            sent = self.nlp(sent)
            for temp in sent.sents:
                
                loc = []
                locTag = []
                for token in nlp(temp.text):
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
                    self.speakers.append(speakers_old[c])
                    
            c += 1
    
        self.CreateMeetingHistogram()    
        
    
    def CreateMeetingHistogram(self, c = 0):
        #computes the local histogram per each speaker
        self.numSpeakers = int(np.max(sp))
        [self.singleWords.append(w) for w in self.wordLemma if w not in self.singleWords]
        self.meetingHisto = np.zeros((Ns+1, len(self.singleWords)))
    
        for w in self.wordLemma:  
            spIdx = sp[c]
            idx   = self.singleWords.index(w)
            self.meetingHisto[0][idx]     += 1
            self.meetingHisto[spIdx][idx] += 1
            c += 1
    
        return [histo, singleWords, Ns]

    def TokenizeReference(self, ref, text = []):
        doc = self.nlp(ref)
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


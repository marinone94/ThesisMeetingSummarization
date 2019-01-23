from PaperTest.config import Config

class Reader(object):
    """Reader contains methods for reading files from the dataset"""
    def __init__(self):
        #config class
        cfg = Config()
        #class parameters from config file
        self.transcriptPath = cfg.transcriptPath
        self.referencePath  = cfg.referencePath
        self.alphabet       = cfg.alphabet
        self.loadHistograms = cfg.loadHistograms
        self.histogramsPath = cfg.histogramsPath
        self.histogramsFile = cfg.histogramsFile
        self.wordsFile      = cfg.wordsFile
        self.histoFolder    = cfg.histoFolder
        self.wordsFolder    = cfg.wordsFolder
        self.numTopics      = cfg.numTopics
        self.topicModelSet1 = cfg.topicModelSet1
        self.topicModelSet2 = cfg.topicModelSet2
        self.topicModelSet3 = cfg.topicModelSet3
        self.topicDict1     = cfg.topicDict1
        self.topicCorpus1   = cfg.topicCorpus1
        self.topicModel1    = cfg.topicModel1
        self.topicDocs1     = cfg.topicDocs1
        self.topicDict2     = cfg.topicDict2
        self.topicCorpus2   = cfg.topicCorpus2
        self.topicModel2    = cfg.topicModel2
        self.topicDocs2     = cfg.topicDocs2
        self.topicDict3     = cfg.topicDict3
        self.topicCorpus3   = cfg.topicCorpus3
        self.topicModel3    = cfg.topicModel3
        self.topicDocs3     = cfg.topicDocs3

    def ReadAll(self):
        return {'Transcripts': self.ReadTranscripts(), 'References': self.ReadReferences(), 'Histograms': self.ReadHistograms(), 'TopicModels': self.LoadTopicModels()}
    
    def ReadTranscripts(self, transcripts = []):
        # read transcripts from the transcript path, if path is valid
        if os.path.exists(self.transcriptPath):
            for file in os.listdir(self.transcriptPath):          
                filename = ''.join([path_tr,file])
                partText = open(filename)
                temp     = self.ReadSingleTranscript(partText.readlines())
                transcripts.append(temp)
         
            return text

        else:
            raise PathError('Transcript path not valid:' + self.transcriptPath)

    def ReadReferences(self, text = []):
        prep = Preprocessing()
        for file in os.listdir(self.referencePath):          

             filename  = ''.join([path ,file])
             partText = open(filename)
             text.append(prep.TokenizeReference(partText.readlines())) #meeting name up to 7th character
        return text
    
    
    def ReadSingleTranscript(self, file, transcript = [], speakers = [], sentences = []):
    
        file = file[3:] #take of first three liness'
        for l in file:
            lVec = l.split().
            speakers.append(self.ConvertLetter(lVec[0]))
            sentences.append(' '.join(lVec[3:]))
            
        merged  = self.MergeSentences(sentences, speakers)
        transcr = [merged['Speakers'], merged['Sentences'][:len(merged['Speakers'])]]
        return transcr

    def ConvertLetter(self, c):
    
        for nums in range(len(self.alphabet)):
            if self.alphabet[nums] == c:
                return nums +1
            else
                raise ImplementationError('SpeakerID not valid')

    def MergeSentences(self, sentences, speakers):
    
        new_sent = []
        new_sp = []
        old_x = 0
        for x in range(1, len(sentences)):
            if sp[x-1]: #when it's zero, new_segment
                if sp[x-1] != sp[x]:
                    new_sp.append(sp[x-1])
                    new_sent.append(' '.join(sent[old_x:x-1]))
                    old_x = x-1
            else:
                new_sp.append(sp[x-1])
                new_sent.append(sent[x-1])
                old_x = x
    
        full_sent = []
        full_sp = []
        y = 0
        for ss in new_sent:
    #        len_ss = len(sk(sen))
            for s in sk(ss):
                full_sent.append(s)
                full_sp.append(new_sp[y])
            y += 1
        return {'Sentences': full_sent, 'Speakers': full_sp}

    def ReadHistograms(self):
        #if loadHistograms is true, read the precomupted ones, else compute them from scratch
        if self.loadHistograms:
            
            [listHistogramsVector, listWordsVector] = self.LoadHistograms()
            arrangedHistogram = [listHistogramsVector, listWordsVector]
            zippedHistograms = self.InvertHistograms(listHistogramsVector, listWordsVector)

            histogram = np.load(''.join([self.histogramsPath, self.histogramsFile]))
            lenDataset = np.sum(histogram)
            lenSingleWords = len(histogram)

            singleWords = self.ReadWordsHistograms(''.join([self.histogramsPath, self.wordsFile]))

            print("HISTOGRAMS LOADED")

            return {'ListHistogramsVector': listHistogramsVector, 'ListWordsVector': listWordsVector, 'ArrangedHistogram': arrangedHistogram, 
                    'ZippedHistograms': zippedHistograms, 'Histogram': histogram, 'LenDataset': lenDataset, 'LenSingleWords': lenSingleWords, 'SingleWords': singleWords]
        
        else:
            raise ImplementationError('Histograms computation not implemented')

    def LoadHisotgrams(self, histo = [], words = []):

        histoPath = ''.join([self.histogramsPath, self.histoFolder])
        wordsPath = ''.join([self.histogramsPath, self.wordsFolder])
        #load histograms
        for file in os.listdir(histoPath):
            filepath = ''.join([histoPath,file])
            histo.append(np.load(filepath))
        #load words for all histograms
        for file in os.listdir(wordsPath):
            filepath = ''.join([wordsPath,file])
            words.append(self.ReadWordsHistograms(filepath))
    
        return histo, words[:-1]

    def ReadWordsHistograms(self, filepath, words = []):
        #all the words are saved in txt files
        txt = open(filepath).readlines()
        [words.append(l[:-1]) for l in txt]
        return words

    def InvertHistograms(self, histos, words, zipped = []):
        [zipped.append([histos[x], words[x]]) for x in range(len(histos))]
        return zipped

    def LoadTopicModels(self, tokensTopicModel = [], topicsDoc = []):
        
        if self.topicModelSet1:
            dictionary = gensim.corpora.Dictionary.load(self.topicDict1)
            corpus = pickle.load(open(self.topicCorpus1, 'rb'))
            ldamodel = gensim.models.ldamodel.LdaModel.load(self.topicModel1)
            with open(self.topicDocs1) as f:
        
        elif self.topicModelSet2:
            dictionary = gensim.corpora.Dictionary.load(self.topicDict2)
            corpus = pickle.load(open(self.topicCorpus2, 'rb'))
            ldamodel = gensim.models.ldamodel.LdaModel.load(self.topicModel2)
            with open(self.topicDocs2) as f:

        else: #error handling in config class 
            dictionary = gensim.corpora.Dictionary.load(self.topicDict3)
            corpus = pickle.load(open(self.topicCorpus3, 'rb'))
            ldamodel = gensim.models.ldamodel.LdaModel.load(self.topicModel3)
            with open(self.topicDocs3) as f:

        for line in f:
            words = line.split()
            if words:
                tokensTopicModel.append(words)
        f.close()
        print('DICTIONARY, CORPUS AND MODEL LOADED!')
        #use gensim pre-built functions to compute topics-terms, topics-docs and vocabulary
        topicTerms = ldamodel.get_topics() 
        [topicsDoc.append(ldamodel.get_document_topics(corpus[x])) for x in range(len(tokensTopicModel))]
        vocab = dictionary.token2id

        return {'Corpus': corpus, 'Ldamodel': ldamodel, 'Vocab': vocab, 'Tokens': tokensTopicModel, 'Docs': topicsDoc, 'Terms': topicTerms, 'Dict': dictionary, 'NumTopics': self.numTopics}







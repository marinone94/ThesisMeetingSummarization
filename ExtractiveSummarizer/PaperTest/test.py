from ExtractiveSummarizer.config import Config

class PaperTest(object):
    """This class is used to replicate and the tests used for the Thesis project"""
    def __init__(self):
        #config class
        cfg              = Config() 
        self.resultPath  = cfg.testResultPath
        #class variables
        self.transcripts = []
        self.references  = []
        self.histograms  = []
        self.topicModels = []
        self.summaries   = []
        self.numMeetings = 0
        
    #read histograms(): return [listHistogramsVector, listWordsVector, arrangedHistogram, zippedHistograms, histogram, lenDataset, lenSingleWords, singleWords]
 
    def TestPaper(self):
        print("Start test ...")
        reader          = Reader()
        evaluation      = Evaluation()
        texts           = reader.ReadAll()
        self.transcripts     = texts['Transcripts']
        self.references      = texts['References']
        self.histograms      = texts['Histograms']
        self.topicModels     = texts['TopicModels']
        print("Datasets and models read ...")
        if self.CheckSizes():
            self.Summarize()
            evaluation.RougeGlobalEvaluation()
            print("Evaluation completed!!!")
        
            return {'Summaries': self.summaries, 'RougeAverages': evaluation.results, 'RougeDeviations': evaluation.stddev}
        else:
            raise ImplementationError('Transcripts and Reference sizes do not match!')
    
    #check that transcript and reference sizes correspond
    def CheckSizes(self):
        return (len(self.transcripts) == len(self.references))
    
    def Summarize(self, i = 1):

        for meeting in self.transcripts:
            print('Meeting' + str(i) + ' ...')
            #preprocessing
            prep = Preprocessing()
            prep.Preprocess(meeting)
            print("Preprocessing completed ...")
            #frequency vectors
            freq = FrequencyMeasures(prep.meetingHisto, prep.singleWords, self.histograms['ListWordsVector'], prep.numSpeakers)
            freq.GetAll()
            print("Frequencies computed ...")
            #functional segmentation
            segm = FuncSegm(prep, freq.suidf, prep.numSpeakers)
            segm.Segmentation()
            print("Segmentation completed ...")
            #keywords
            keyw = Extractor(prep, segm, freq.idf)
            keyw.ExtractKeywords()
            print("Keywords extracted ...")
            #check if monologue or dialogue and apply specific method
            localSummary = []
            for dstr in segm.speakerDistr:
                if (np.sum(dstr) == 1):
                    mon = Monologue()
                    mon.Summarize()
                    localSummary.append(mon.summary)
                    print("Monologue summarized ...")
                else:
                    dial = Dialogue()
                    dial.Summarize()
                    localSummary.append(dial.summary)
                    print("Dialogue summarized ...")
            #join, save and append the final summary
            txtSummary = ' '.join(localSummary)
            Help.SaveFileTxt(txtSummary, str(i), self.resultPath)
            i += 1
            self.summaries.append(txtSummary)
            print("Summary stored ...")
        
        print("Dataset summarized!!!")



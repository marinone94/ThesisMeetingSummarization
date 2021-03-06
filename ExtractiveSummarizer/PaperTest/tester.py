from config import Config
from Reader.reader import Reader
from Evaluation.rougeEvaluation import RougeEvaluation as Evaluation
from Preprocessing.preprocessing import Preprocessing
from FrequencyMeasures.frequencymeasures import FrequencyMeasures
from FunctionalSegmentation.funcsegm import FuncSegm
from ExtractKeywords.extractor import Extractor
from MonologueSummarizer.monologue import Monologue
from DialogueSummarizer.dialogue import Dialogue
from PaperTest.help import Help
import numpy as np

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
        
        texts           = reader.ReadAll()
        self.transcripts     = texts['Transcripts']
        self.references      = texts['References']
        self.histograms      = texts['Histograms']
        self.topicModels     = texts['TopicModels']
        print("Datasets and models read ...")
        if self.CheckSizes():
            self.Summarize()
            evaluation      = Evaluation(self.summaries, self.references)
            evaluation.RougeGlobalEvaluation()
            print("Evaluation completed!!!")
        
            return {'Summaries': self.summaries, 'RougeAverages': evaluation.results, 'RougeDeviations': evaluation.stddev}
        else:
            raise NotImplementedError('Transcripts and Reference sizes do not match!')
    
    #check that transcript and reference sizes correspond
    def CheckSizes(self):
        return (len(self.transcripts) == len(self.references))
    
    def Summarize(self, x = 1):

        for meeting in self.transcripts:
            print('\n\n\n\n\nMeeting ' + str(x) + ' ...')
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
            i = 1
            for dstr in segm.speakerDistr:
                if len(segm.cleanSentences[i-1]) > 1:
                    if (np.sum(dstr) == 1):
                        mon = Monologue(segm, keyw, i-1)
                        mon.Summarize()
                        localSummary.append(mon.summary)
                        print("Monologue summarized ...")
                    else:
                        dial = Dialogue(prep, segm, self.histograms, self.topicModels, freq.suidf, freq.tfidfSpeak, i-1)
                        dial.Summarize()
                        localSummary.append(dial.summary)
                        print("Dialogue summarized ...")
                elif len(segm.cleanSentences[i-1]) == 1:
                    localSummary.append(str(segm.cleanSentOrig[i-1]))
                else:
                    ...
                i += 1
            
            #join, save and append the final summary
            txtSummary = ' '.join(localSummary)
            Help.SaveFileTxt(txtSummary, 'summary_' + str(x), self.resultPath)
            x += 1
            self.summaries.append(txtSummary)
            print("Summary stored ...")
        
        print("Dataset summarized!!!")



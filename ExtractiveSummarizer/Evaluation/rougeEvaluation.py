class RougeEvaluation(object):
    """It computes the metrics used in the Thesis project"""
    def __init__(self, summaries, references):
        #global config
        cfg = Config()
        #class variables
        self.f1  = []
        self.p1  = []
        self.r1  = []
        self.f2  = []
        self.p2  = []
        self.r2  = []
        self.f_l = []
        self.p_l = []
        self.r_l = []
        #summaries and references
        self.summaries      = summaries
        self.references     = references
        #global results
        self.results        = []
        self.stddev         = []
        #folder destination path
        self.testResultPath = cfg.testResultPath
        avgResultFile       = cfg.avgResultFile
        stdResultFile       = cfg.stdResultFile

    def RougeGlobalEvaluation(self):
        for (summary, reference) in zip(self.summaries, self.references):
            self.RougeSingleEvaluation(summary, reference)
        self.TestResults()
    
    def RougeSingleEvaluation(self, summary, reference):
        rouge_score = rouge.get_scores(text_summary, ref)[0]
        self.f1.append(rouge_score["rouge-1"]["f"])
        self.p1.append(rouge_score["rouge-1"]["p"])
        self.r1.append(rouge_score["rouge-1"]["r"])
        self.f2.append(rouge_score["rouge-2"]["f"])
        self.p2.append(rouge_score["rouge-2"]["p"])
        self.r2.append(rouge_score["rouge-2"]["r"])
        self.f_l.append(rouge_score["rouge-l"]["f"])
        self.p_l.append(rouge_score["rouge-l"]["p"])
        self.r_l.append(rouge_score["rouge-l"]["r"])

    def TestResults(self):
        self.results = [np.average(self.f1), np.average(self.p1), np.average(self.r1), 
                   np.average(self.f2), np.average(self.p2), np.average(self.r2), 
                   np.average(self.f_l), np.average(self.p_l), np.average(self.r_l)]
        self.stddev  = [np.std(self.f1), np.std(self.p1), np.std(self.r1), 
                   np.std(self.f2), np.std(self.p2), np.std(self.r2), 
                   np.std(self.f_l), np.std(self.p_l), np.std(self.r_l)]

        np.save(''.join([self.testResultPath, self.avgResultFile]), self.results)
        np.save(''.join([self.testResultPath, self.stdResultFile]), self.stddev)

        print(self.results)
        print(self.stddev)

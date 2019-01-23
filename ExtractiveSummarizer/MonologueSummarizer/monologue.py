class Monologue(object):
    """description of class"""
    def __init__(self, segm, keyw):
        #config class
        cfg = Config()
        #class viarables
        self.segm       = segm
        self.keyw       = keyw
        self.ratio      = cfg.monologueRatio
        self.lpAddress  = cfg.lpAddress
        #class results
        self.summary    = []
        self.optS       = []
    
    def Summarize(self):
        self.Optimize()
        self.ExtractSummary()

    def Optimize(self, c = 0, s = 0, num_words = 0)
        num_sent = len(self.segm.cleanSentences)
        summary = []
    
        o = create_o(self.keyw.keywords, self.segm.cleanSentences)
        l = np.zeros(len(self.segm.cleanSentences), dtype=int) #length of ith utterance
        for x in range(0, len(l)):
            l[x] = len(self.segm.cleanSentences[x])
            num_words = num_words + l[x]

        L = int(num_words * self.ratio) #length constraint
        prob = LpProblem("extract monlogues_spacy", LpMaximize)

        for i in range(len(self.keyw.keywords)):
            c.append(LpVariable(name='c_%s' %i, cat='Binary'))
        for j in range(num_sent):
            s.append(LpVariable(name='s_%s' %j, cat='Binary'))

        obj = lpSum([self.keyw.keywScores[i] * c[i] for i in range(len(self.keyw.keywords))])
        prob += obj

        sum_jl = lpSum([np.multiply(s[j], l[j]) for j in range(num_sent)])
        prob += sum_jl <= L #constraints
        for i in range(len(self.keyw.keywords)):
            sum_jo = lpSum([np.multiply(s[j], o[i][j]) for j in range(num_sent)])  
            prob += sum_jo >= c[i]
    
        for i in range(len(self.keyw.keywords)):
            for j in range(num_sent):
                prob += np.multiply(s[j], o[i][j]) <= c[i]
                #o and o.T are the same matrix , just transposed, impose it
        prob.writeLP(self.lpAddress)
        sol = prob.solve()
        for i in range(num_sent):
            self.optS.append(value(s[i]))


    def ExtractSummary(self, localSummary = []):
        for x in range(0, len(self.optS)):
            if self.optS[x] and self.segm.cleanSentOrig[x] not in localSummary:
                localSummary.append(self.segm.cleanSentOrig[x])
            else:
                localSummary.append('')

        self.summary = ' '.join(localSummary)



   
class Help(object):
    """Help class for reader methods"""

    def ConvertLetter(alphabet, c):
        if c not in alphabet:
            raise ValueError('SpeakerID not valid')
        for nums in range(len(alphabet)):
            if alphabet[nums] == c:
                return nums +1
        return nums





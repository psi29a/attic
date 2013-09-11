#!/usr/bin/env python

LAST_LETTER_CANDIDATES_MAX = 5;

initialized = False

names = ['Andor', 'Baatar','Drogo', 'Grog', 'Gruumsh', 'Grunt', 'Hodor',
            'Hrothgar', 'Hrun', 'Korg', 'Lothar', 'Odin', 'Thor', 'Yngvar',
            'Xandor']
sizes = []

letters = {}
firstLetterSamples = []
lastLetterSamples = []


# Weighted Letter Counter. Adds counting functionality to a letter.
# @author Ebyan Alvarez-Buylla
class WeightedLetterCounter:
    def __init__(self, p_letter):
        self.letter = p_letter
        self.count = 0

class WeightedLetterGroup:
    def __init__(self):
        self.letters = {}
        self.letterSamples = []

    def add(self, letter):
        if letter not in self.letters:
            self.letters[letter] = WeightedLetterCounter(letter);

        self.letters[letter].count += 1
    # Expand the letters dictionary into an array, for ease of picking letters
    # in the main body loop. We still keep the letters data for more easily
    # finding a best-fit.
    def expandSamples(self):
        for letter in self.letters:
            for i in range(0, self.letters[letter].count):
                self.letterSamples.append(self.letters[letter].letter)

# WeightedLetter. Holds the letter distribution information.
# @author Ebyan Alvarez-Buylla
class WeightedLetter:
    def __init__(self, p_letter):
        self.letter = p_letter
        self.nextLetters = WeightedLetterGroup()

    # Have the WeightedLetterGroup keep track of the next letter instead of a simple
    # array. A simple array will do the trick (removing the need for WightedLetterGroup
    # or WeightedLetterCounter), if you use a different algorithm for best-fitting
    # the penultimate letter (see WeightedLetterNamegen.getIntermediateLetter()).
    # @param    nextLetter
    def addNextLetter(self, nextLetter):
        self.nextLetters.add(nextLetter)


#
# Author: Michael Homer
# Date: Sunday, April 26th, 2009
# License: MIT
#
def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]



# Searches for the best fit letter between the letter before and the letter after (non-random).
# Used to determine penultimate letters in names.
# @param    letterBefore    The letter before the desired letter.
# @param    letterAfter        The letter after the desired letter.
# @return    The best fit letter between the provided letters.

def getIntermediateLetter(letterBefore, letterAfter):
    if letterBefore and letterAfter:
        # First grab all letters that come after the 'letterBefore'
        letterCandidates = letters[letterBefore].nextLetters.letters

        bestFitLetter = None
        bestFitScore = 0

        # Step through candidates, and return best scoring letter
        for letter in letterCandidates:
            weightedLetterGroup = letters[letter].nextLetters
            letterCounter = weightedLetterGroup.letters.get(letterAfter, None)
            if letterCounter:
                if letterCounter.count > bestFitScore:
                    bestFitLetter = letter
                    bestFitScore = letterCounter.count
        return bestFitLetter;
    else:
        # If any of the passed parameters were null, return null. This happens when the letterBefore has no candidates.
        return None


# Checks that no three letters happen in succession.
# @param    name    The name array (easier to iterate)
# @return    True if no triple letter sequence is found.
def tripleLetterCheck(name):
    for i in range(2,len(name)):
        if ( name[i] == name[i - 1] and name[i] == name[i - 2] ):
            return False
    return True

# Checks that the Damerau-Levenshtein distance of this name is within a 
# given bias from a name on the master list.
# @param    name    The name string.
# @return    True if a name is found that is within the bias.
def checkLevenshtein(name):
    import sys

    levenshteinBias = len(name)/2

    # Grab the closest matches, just for fun
    closestName = ""
    closestDistance = sys.maxsize

    for i in range(0,len(names)):
        levenshteinDistance = dameraulevenshtein(name, names[i])

        # This is just to get an idea of what is failing
        if levenshteinDistance < closestDistance:
            closestDistance = levenshteinDistance
            closestName = names[i]

        if levenshteinDistance <= levenshteinBias:
            return True
    return False

def main():
    from sys import argv
    import argparse
    import timeit
    import random

    #print ("Within DL range: ", checkLevenshtein("test"))
    #print ("L: ", levenshtein("time","tiem"))
    #print ("L: ", timeit.timeit("levenshtein('e0zdvfb840174ut74j2v7gabx1 5bs', 'qpk5vei 4tzo0bglx8rl7e 2h4uei7')", setup="from __main__ import levenshtein", number=100), " seconds")
    #print ("DL:", dameraulevenshtein("time","tiem"))
    #print ("DL:",timeit.timeit("dameraulevenshtein('e0zdvfb840174ut74j2v7gabx1 5bs', 'qpk5vei 4tzo0bglx8rl7e 2h4uei7')", setup="from __main__ import dameraulevenshtein", number=100), " seconds")

    ######### init ##########
    for i in range(0, len(names)):
        name = names[i]

        # (1) Insert size
        sizes.append(len(name))

        # (2) Grab first letter
        firstLetterSamples.append(name[0])

        # (3) Grab last letter
        lastLetterSamples.append(name[len(name) - 1])

        # (4) Process all letters
        for n in range(0, len(name)-1):
            letter = name[n]
            nextLetter = name[n + 1]

            # Create letter if it doesn't exist
            if not letter in letters:
                letters[letter] = WeightedLetter(letter)

            letters[letter].addNextLetter(nextLetter)

            # If letter was uppercase (beginning of name), also add a lowercase entry
            if letter != letter.lower():
                letter = letter.lower()

                # Create letter if it doesn't exist
                if not letter in letters:
                    letters[letter] = WeightedLetter(letter)

                letters[letter].addNextLetter(nextLetter)

    # Expand letters into samples
    for weightedLetter in letters:
        letters[weightedLetter].nextLetters.expandSamples();

    initialized = True


    ######### generate ##########
    # Initialize if called for the first time
    if not initialized:
        return

    amountToGenerate = 10
    result = [];

    for nameCount in range(0, amountToGenerate):
        name = []

        # Pick size
        size = random.choice(sizes)

        # Pick first letter
        firstLetter = random.choice(firstLetterSamples)
        name.append(firstLetter);

        for i in range(1, size-1):
            # Only continue if the last letter added was non-null
            if name[i-1]:
                weightedLetter = letters[name[i - 1]]
                name.append(random.choice(weightedLetter.nextLetters.letterSamples))
            else:
                break

        # Attempt to find a last letter
        for lastLetterFits in range(0, LAST_LETTER_CANDIDATES_MAX):
            lastLetter = random.choice(lastLetterSamples)
            intermediateLetterCandidate = getIntermediateLetter(name[len(name) - 1], lastLetter)

            # Only attach last letter if the candidate is valid
            # if no candidate, the antepenultimate letter always occurs at the end
            if intermediateLetterCandidate:
                name.append(intermediateLetterCandidate)
                name.append(lastLetter)
                break

        nameString = "".join(name)

        # Check that the word has no triple letter sequences
        # and that the Levenshtein distance is withen an accepted bias
        if tripleLetterCheck(name) and checkLevenshtein(nameString):
            result.append(nameString)
            # Only increase the counter if we've successfully added a name
            nameCount += 1

    print ("Result: ", result)
    #return result


if __name__ == '__main__':
    #import cProfile
    #cProfile.run('ex = main()')
    main()

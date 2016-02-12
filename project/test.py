###########Capital Letter words##############################
unigrams = {}
a = "This is Krishna Chaitanya Sripada studying in University of Colorado boulder"
finalWord = ""
for word in a.split()[1:]:
    if not word.islower():
        finalWord=finalWord +" "+word
    else:
        if len(finalWord)>0:
            if finalWord in unigrams.keys():
                unigrams[finalWord.lstrip()]+=1
            else:
                unigrams[finalWord.lstrip()]=1
            finalWord = ""

##########Adding the last one if any#################
if len(finalWord)>0:
    if finalWord in unigrams.keys():
        unigrams[finalWord.lstrip()]+=1
    else:
        unigrams[finalWord.lstrip()]=1

print unigrams

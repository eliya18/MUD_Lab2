#! /usr/bin/python3

import sys
import re
from os import listdir
from collections import Counter
from nltk import pos_tag
# nltk.download('averaged_perceptron_tagger')
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
import csv


# --------- load HSDB.txt-----------
# -- Load external resource into a set

hsdb_set = set()
with open('resources/HSDB.txt', 'r') as f:
    for line in f:
        hsdb_set.add(line.strip())
   
## --------- load DrugBank.txt into a dictionary ----------- 
drugs = {}
with open('resources/DrugBank.txt', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='|')
    for row in reader:
        drug_name = row[0].lower()
        drug_category = row[1]
        if drug_name not in drugs:
            drugs[drug_name] = drug_category
   


## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens) :

   # for each token, generate list of features and add it to the result
   result = []
   words_list = []
   n = 2
   for i in tokens:
      words_list.append(i[0])
   pos_tags = pos_tag(words_list)
   for k in range(0,len(tokens)):
      tokenFeatures = [];
      t = tokens[k][0]
      tokenFeatures.append("form="+t)
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("suf4="+t[-4:])
      tokenFeatures.append("suf5=" + str(t[-5:]))
      tokenFeatures.append("pre3="+ t[:3])
      tokenFeatures.append("pre4=" + t[:4])
      tokenFeatures.append("pre5=" + t[:5])
      tokenFeatures.append("length=" + str(len(t)))
      tokenFeatures.append("ngrams=" + str(Counter([t])))
      tokenFeatures.append("posTags=" + str(pos_tags[k][1]))
      tokenFeatures.append("containNumber=" + str(bool(re.search(r'\d', t))))
      tokenFeatures.append("hasDash=" + str(bool(re.search('-', t))))  # True if token contains a dash
      tokenFeatures.append("allUpper=" + str(t.isupper()))  # True if all characters in token are uppercase
      tokenFeatures.append("isTitle= " + str(bool(re.match(r'[A-Z].[a-z]*', t))))
      # tokenFeatures.append("isFistUpper="+ str(bool(re.match(r'^[A-Z]',t))))
      tokenFeatures.append("isFirstUpper=" + str(t[0].isupper()))
      # tokenFeatures.append("isCamelCase=" + str(bool(re.match(r'^[a-z]+([A-Z][a-z]+)+$', t)))) #F1 decrease

      # # Add the feature of looking up drug names in DrugBank.txt
      if t.lower() in drugs:
         tokenFeatures.append("InDrugBank=True")
         drug_category = drugs[t.lower()]
         tokenFeatures.append("type=" + drug_category)
      else:
         tokenFeatures.append("InDrugBank=False")
      if t in hsdb_set:
         tokenFeatures.append("inHSDB=True")
      else:
         tokenFeatures.append("inHSDB=False")


      if k>0 :
         tPrev = tokens[k-1][0]
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         # tokenFeatures.append("suf4Prev=" + tPrev[-4:])
         # tokenFeatures.append("suf5Prev=" + tPrev[-5:])
         tokenFeatures.append("lengthPrev=" + str(len(tPrev)))
         #tokenFeatures.append("posTagsPrev=" + str(pos_tags[k-1][1]))
      else :
         tokenFeatures.append("BoS")

      if k<len(tokens)-1 :
         tNext = tokens[k+1][0]
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("suf3Next="+tNext[-3:])
         # tokenFeatures.append("suf4Next=" + tNext[-4:])
         # tokenFeatures.append("suf5Next=" + tNext[-5:])
         tokenFeatures.append("lengthNext=" + str(len(tNext)))
         #tokenFeatures.append("posTagsNext=" + str(pos_tags[k+1][1]))
      else:
         tokenFeatures.append("EoS")
    
      result.append(tokenFeatures)
    
   return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      # print(tokens)
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()

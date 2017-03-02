import re
import csv
import random


def posPre(word, preps):
    for prep in preps:
        if word.find(prep) >= 0:
            return 1

    return 0

def negPre(word,names):
    for name in names:
        if word.find(name) >= 0:
            return -2

    return 0

def containNeg(word, names):
    for name in names:
        if word.find(name) >= 0:
            return -2

    return 0


def Captalized(word):
    for str in word.split(" "):
        if not str[0].isupper():
            return 0
    return 1

def containNum(word):
    str = ''.join(word.split(' '))
    if str.isalpha():
        return 0
    return 1


#contain negative words before
def containNegs(word, negsbefore):
    for neg in negsbefore:
        if word.find(neg) >= 0:
            return -3
    return 0


    

names = ['in','from','on','Chef','chef', 'Blvd', 'L.A.', 'LA', 'Los Angeles', 'Beverly','Hills']
preps = ['to','at']


def getvector(ii,c):
    file = '%s%d%s' %('doc/', ii, '.txt')
    f = open(file)
    pre_words = []
    instance = []
    labels = []
    reg = r'\[([^\[\]]+)\]|\{([^\{\}]+)\}'
    for line in f:
        strs = re.findall(reg, line)
        for str in strs:
            if str[0] == '':
                instance.append(str[1].strip())
                labels.append(1)
            else:
                instance.append(str[0].strip())
                labels.append(0)
        words = re.split('[ .-]',line)
        for i in range(len(words)):
            if len(words[i]) == 0:
                continue
            first = words[i][0]

            if  words[i].find('{') >=0 or words[i].find('[') >=0:
                if i == 0:
                    pre_words.append(" ")
                else:
                    pre_words.append(words[i-1].lower())

    res = []

    sz = len(instance)
    for i in range(sz):
            res.append('%d%s' %(ii,'.txt'))
            res.append(instance[i])
            
            res.append(posPre(pre_words[i],preps)) 
            
            res.append(negPre(pre_words[i],names)) 
            
            res.append(containNeg(instance[i],names)) 
            
            res.append(len(instance[i])) 
            
            res.append(Captalized((instance[i]))) 
            
            res.append(len(instance[i].split(" "))) 
            
            res.append(containNum(instance[i]))
            
            res.append(labels[i])
            c.writerow(res)
            res = []



if __name__ == '__main__':
    list = range(1,314)
    random.shuffle(list)
    header = ['filename','instance','Positive Prewords','Negative Prewords', 'Contain Negative', 'len of char','All capatalized'
                   , 'len of words','including digit','labels']
    with open('whole_set.csv','wb') as csvfile:
         spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

         spamwriter.writerow(header)
         for i in range(1,314):
             getvector(i,spamwriter)
    with open('trainset.csv', 'wb') as csvfile:
         spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

         spamwriter.writerow(header)
         for i in range(1,200):
             getvector(list[i],spamwriter)
    with open('testset.csv', 'wb') as csvfile:
         spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
         spamwriter.writerow(header)
         for i in range(200,313):
             getvector(list[i],spamwriter)

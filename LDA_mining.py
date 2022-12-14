import spacy
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from gensim.models import HdpModel
import pandas as pd
import re
from textblob import TextBlob

# helper function to split sentences
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
# import spacy for keywords set-dimension mapping
nlp = spacy.load("en_core_web_sm")
def semantic_similarity(cur_str, current_keyword):
    cur_str += current_keyword
    tokens = nlp(cur_str)
    token1, token2 = tokens[0:len(tokens) - 1], tokens[len(tokens) - 1]
    return token1.similarity(token2)






# Prepare keyword set
# Food
food_keywords = []
food_str =""
# service
service_keywords = []
service_str =""
# enviornment
environment_keywords = []
environment_str =""
# preference matrix
preference_matrix = []
# sentiment matrix
sentiment_matrix = []
# rating matrix(each dimension)
rating_matrix = []

# import food.txt
with open('food.txt') as f:
    food_keywords = [line.strip('\n') for line in f]
# construct food str
for each in food_keywords:
    food_str += each
    food_str += " "

# import environment txt
with open('env.txt') as f:
    environment_keywords = [line.strip('\n') for line in f]
for each in environment_keywords:
    environment_str += each
    environment_str += " "

# import service.txt
with open('service.txt') as f:
    service_keywords = [line.strip('\n') for line in f]
for each in service_keywords:
    service_str += each
    service_str += " "

# Read CSV file
data = pd.read_csv('GoogleMap.csv')
current_reviews =[]
# for Broadway Oyster Bar
for i in range(0,len(data)):
    if data['restaurant'][i]=='Broadway Oyster Bar':
        current_reviews.append([i, data['score'][i], data['sentence'][i]])
print(len(current_reviews))

# from nltk.corpus import stopwords
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents
# doc_a = "Mike took great care of us @ the outside bar!! He was terrific! The oysters are fresh, the gumbo is lovely, the wings were very tasty! The eclectic nature of Broadway Oyster Bar only adds to you enjoyment. There is something to see every square inch!  We had s great time!"
# doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
# doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
# doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
# doc_e = "Health professionals say that brocolli is good for your health."

# for each review in the review set
for index in range(0,len(current_reviews)):
    doc_a = current_reviews[index][2]
    # compile sample documents into a list
    doc_set = [doc_a]

    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)

    print(type(texts))
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
    # print(ldamodel.print_topics(num_topics=3, num_words=6))

    # Use the HDP value to mine the dimension, top 3 and compare it with the keyword set to get the frequency matrix
    hdp = HdpModel(corpus=corpus, id2word=dictionary)
    # Get the HDP to limit the number of dimension
    alpha = hdp.hdp_to_lda()[0]
    # print(alpha)

    # ranked based on alpha values, most significant topics
    # print(hdp.print_topics(num_topics=3))
    # print(hdp.print_topics(num_topics=3)[0][1].split('+'))
    # average the topic(select top 3 most significant) and normalize to get preference matrix
    food_prob = 0
    service_prob = 0
    environment_prob = 0
    for i in range(0, 1):
        cur_topic_str = hdp.print_topics(num_topics=3)[i][1].split('+')
        tuples = []
        # iterate through the str and store frequency-words as tuple
        for tuple in cur_topic_str:
            tuples.append([tuple.split('*')[0], tuple.split('*')[1]])
        # now that we have all the tuples [0]--fre  [1]---word, map the frequency to each dimension
        for t in tuples:
            food_similarity = semantic_similarity(food_str, t[1])
            service_similarity = semantic_similarity(service_str, t[1])
            environment_similarity = semantic_similarity(environment_str, t[1])
            # select the max similar
            if max(food_similarity,service_similarity,environment_similarity)==food_similarity:
                food_prob += float(t[0])
            elif max(food_similarity,service_similarity,environment_similarity)==service_similarity:
                service_prob += float(t[0])
            else:
                environment_prob += float(t[0])
    # done getting frequency from top 3 topics, normalize the 3 distribution
    total = food_prob+service_prob+environment_prob
    food_dist = food_prob/total
    service_dist = service_prob/total
    environment_dist = environment_prob/total
    # add to preference matrix
    preference_matrix.append([food_dist,service_dist,environment_dist])
    # print(len(preference_matrix))

    # break down each review into sentences and calculate polarity, allocate the polarity
    food_sentiment =0
    service_sentiment=0
    environment_sentiment=0
    # split the review
    sentence_block = split_into_sentences(doc_a)
    # for each words of the review, calculate relavance to dimension and assign the polarity
    for sen in sentence_block:
        blb = TextBlob(sen)
        tp_polarity = blb.polarity
        # tokenize each sentence and get the dimension
        cur_texts = []
        # clean and tokenize document string
        raw = sen.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        cur_texts.append(stemmed_tokens)
        # calculate the frequency in each sentence and assign the polarity according to final probability distribution
        sen_food_prob =0
        sen_service_prob=0
        sen_environment_prob=0

print(preference_matrix)




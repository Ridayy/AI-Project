import pandas as pd
import numpy as np
import sys
import nltk
from sklearn.preprocessing  import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier


def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


def load_data_set(set):
    return pd.read_table(set, header=None, encoding='utf-8')


def optimize_messages(text_messages):
    processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')
    processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                    'webaddress')

    processed = processed.str.replace(r'£|\$', 'moneysymb')
        

    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                    'phonenumbr')
        
    processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

    processed = processed.str.replace(r'[^\w\d\s]', ' ')

    processed = processed.str.replace(r'\s+', ' ')

    processed = processed.str.replace(r'^\s+|\s+?$', '')

    processed = processed.str.lower()

    return processed


def pre_process_data(df):
    classes = df[0]
    encoder = LabelEncoder()
    return encoder.fit_transform(classes)


def remove_stopwords(processed):
    stop_words = set(stopwords.words('english'))
    processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))
    return processed


def remove_wordstems(processed):
    ps = nltk.PorterStemmer()
    processed = processed.apply(lambda x: ' '.join(
    ps.stem(term) for term in x.split()))
    return processed


def create_words(processed):
    all_words = []

    for message in processed:
        words = word_tokenize(message)
        for w in words:
            all_words.append(w)
            
    all_words = nltk.FreqDist(all_words)

    return all_words


df = load_data_set('SMSSpamCollection')

Y = pre_process_data(df)

text_messages = df[1]
# print(text_messages[:10])

processed = optimize_messages(text_messages)

processed = remove_stopwords(processed)

processed = remove_wordstems(processed)

all_words = create_words(processed)

word_features = list(all_words.keys())[:1500]

features = find_features(processed[0])

messages = list(zip(processed, Y))

seed = 1
np.random.seed = seed
np.random.shuffle(messages)

featuresets = [(find_features(text), label) for (text, label) in messages]


training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]


models = list(zip(names, classifiers))
print(models)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))



nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))

txt_features, labels = zip(*testing)
prediction = nltk_ensemble.classify_many(txt_features)

print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])
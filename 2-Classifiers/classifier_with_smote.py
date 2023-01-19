import nltk
#from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

nltk.download('stopwords')
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics, model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import scipy as sp
from scipy import special
import re
import scipy.sparse

dataset_train = pd.read_csv('data/senti_train_imbalanced.csv',encoding='latin-1')
dataset_test= pd.read_csv('data/senti_test.csv',encoding='latin-1')
# dtf.head()


dataset_train['review'] = dataset_train['review'].fillna('').apply(str)
dataset_test['review'] = dataset_test['review'].fillna('').apply(str)


# #Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text= nltk.re.sub(pattern, '', text)
    return text
#Apply function on review column
dataset_train['review']=dataset_train['review'].apply(remove_special_characters)
dataset_test['review']=dataset_test['review'].apply(remove_special_characters)


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return nltk.re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
dataset_train['review']=dataset_train['review'].apply(denoise_text)
dataset_test['review']=dataset_test['review'].apply(denoise_text)



def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
dataset_train['review']=dataset_train['review'].apply(simple_stemmer)
dataset_test['review']=dataset_test['review'].apply(simple_stemmer)


#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)
#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
#Apply function on review column
dataset_train['review']=dataset_train['review'].apply(remove_stopwords)
dataset_test['review']=dataset_test['review'].apply(remove_stopwords)


vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, lowercase=False)
x_train_conted = vectorizer.fit_transform(dataset_train['review'])
x_test_conted = vectorizer.fit_transform(dataset_test['review'])

dtf1_train= pd.DataFrame(x_train_conted.toarray(), columns=vectorizer.get_feature_names())
dataset_train.drop('review', axis=1, inplace=True)
res_train = pd.concat([dataset_train, dtf1_train], axis=1)

dtf1_test= pd.DataFrame(x_test_conted.toarray(), columns=vectorizer.get_feature_names())
dataset_test.drop('review', axis=1, inplace=True)
res_test = pd.concat([dataset_test, dtf1_test], axis=1)


print(res_train)
print("_______________________________________________")
print(res_test)

X_train = res_train.loc[:, res_train.columns != 'sentiment'].values
y_train = res_train.loc[:, 'sentiment'].values

X_test = res_test.loc[:, res_test.columns != 'sentiment'].values
y_test = res_test.loc[:, 'sentiment'].values

sm = SMOTE()
sm_X_train_vec, sm_train_y = sm.fit_sample(X_train, y_train)

df = pd.concat([pd.DataFrame(sm_X_train_vec), pd.DataFrame(sm_train_y)], axis=1)
df.to_csv('df_smoted_nb.csv', index=False, encoding='utf-8')


print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


print(sm_X_train_vec.shape,sm_train_y.shape)

# Random Forest  
# clf=RandomForestClassifier()

# Naive Bayes
# clf=MultinomialNB()

# Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(sm_X_train_vec,sm_train_y)
predicted = clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))
print(confusion_matrix(y_test, predicted))


tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0,1]).ravel()
specificity = tn / (tn+fp)
print("specificity is")
print(specificity)

# Accuracy
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, predicted)
print("accuracy_score is")
print(acc)

# Recall
from sklearn.metrics import recall_score
recall=recall_score(y_test, predicted, average=None)
print("recall_score is")
print(recall)

# Precision
from sklearn.metrics import precision_score
precision=precision_score(y_test, predicted, average=None)
print("precision_score is")
print(precision)

from sklearn.metrics import f1_score
f1=f1_score(y_test, predicted, average=None)
print("f1_score is")
print(f1)
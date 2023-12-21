import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from spacytextblob.spacytextblob import SpacyTextBlob
from sklearn.model_selection import train_test_split
import pickle
from sklearn.naive_bayes import GaussianNB
import spacy
import nltk
from spacy.lang.en import stop_words, punctuation
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from rich.console import Console
nltk.download('vader_lexicon')

console = Console() 
nlp=spacy.load("en_core_web_lg")

sa_nlp=spacy.load("en_core_web_lg")
sa_nlp.add_pipe("spacytextblob")

def pre_process(review_txt):
    review_txt_processed=""
    doc=nlp(review_txt)
    review_txt_processed = ' '.join([token.lemma_.lower() for token in doc if not(token.is_punct and token.is_digit)])
    review_txt_processed = review_txt_processed[:200]
    return review_txt_processed


# console.print(stop_words.STOP_WORDS)


samples_df = pd.read_csv('sample_train.csv', encoding="utf-8", header=0)
samples_test=pd.read_csv('sample_test.csv', encoding="utf-8", header=0)

console.print(f"[white on green] Preparing Dataframe ...")
samples_df['all_text'] = samples_df['summary'] + ' ' + samples_df['text']
samples_test['all_text'] = samples_test['summary'] + ' ' + samples_test['text']
# samples_df=df.groupby('rate').sample(5000)
# samples_test=df_test.groupby('rate').sample(1000)
# samples_df.to_csv("sample_df",'wb')
# samples_test.to_csv("samples_test",'wb')
sia = SentimentIntensityAnalyzer()
res = {}
counter=0
for i in range(samples_df.shape[0]):
    text = samples_df.iloc[i]['all_text']
    text = pre_process(str(text))
    res["sia"] = sia.polarity_scores(text)
    res["textblob"] = sa_nlp(text)._.polarity
    counter+=1
    print(counter)
    # console.print(f"[white on red]{df.iloc[i]['rate']}:[/] | [yellow]{res['sia']}[/] | [green]{res['textblob']}[/] {df.iloc[i]['all_text']}\n [yellow]{text}")


samples_df['all_text']=samples_df['all_text'].apply(pre_process)
# print(len(samples_df.iloc[0]['all_text']),len(samples_df.iloc[10]['all_text']))

# X_train, X_test, y_train, y_test = train_test_split(df.all_text, df.rate, test_size=0.2,train_size=9,stratify=df.rate)
X_train=samples_df['all_text']
y_train=samples_df['rate']
X_test=samples_test['all_text']
y_test=samples_test['rate']

vecotrizer = TfidfVectorizer()

X_train_count = vecotrizer.fit_transform(X_train.values)
X_test_count = vecotrizer.transform(X_test.values)
model = GaussianNB()
X_train_count=X_train_count.toarray()
X_test_count=X_test_count.toarray()

# Train the model on the training data.
model.fit(X_train_count, y_train)
print(model.score(X_test_count,y_test))

# Make predictions on the test data.
y_pred = model.predict(X_test_count)

file_name = 'modelidf.pkl'

# Pickle the model.
with open(file_name, 'wb') as f:
    pickle.dump(model, f)

# Close the file.
f.close()

# Evaluate the model on the test data.
print("\n\n",classification_report(y_test, y_pred))



predictions=model.predict(X_test_count)
cm = confusion_matrix(predictions, y_test, labels=model.classes_)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

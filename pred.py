import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from spacytextblob.spacytextblob import SpacyTextBlob
from sklearn.model_selection import train_test_split
import pickle
from sklearn.naive_bayes import GaussianNB
import spacy
# import nltk
from spacy.lang.en import stop_words, punctuation
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from rich.console import Console
import numpy as np
# nltk.download('vader_lexicon')

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

# df = pd.read_csv('train.csv', encoding="utf-8", header=0)
samples_test=pd.read_csv('reviews_9records.csv', encoding="utf-8", header=0)

console.print(f"[white on green] Preparing Dataframe ...")
# df['all_text'] = df['summary'] + ' ' + df['text']
samples_test['all_text'] = samples_test['summary'] + ' ' + samples_test['text']
# samples_df=df.groupby('rate').sample(200)
# samples_test=df_test.groupby('rate').sample(1000)
# samples_df.to_csv("sample_df",'wb')
# samples_test.to_csv("samples_test",'wb')

sia = SentimentIntensityAnalyzer()
lst = []
counter = 0
for i in range(samples_test.shape[0]):
    text = samples_test.iloc[i]['all_text']
    text = pre_process(str(text))
    
    res = {}  
    res["sia"] = sia.polarity_scores(text)
    res["textblob"] = sa_nlp(text)._.polarity
    
    lst.append(res)  
    counter += 1
    print(counter)

lst_of_sia = []
lst_of_text=[]
for item in lst:
    # sia_scores = [item["sia"]["neg"], item["sia"]["neu"], item["sia"]["pos"], item["sia"]["compound"]]
    sia_scores = float(item ["sia"]["compound"])
    if sia_scores<0.2:
        sia_scores=1
    else :
        sia_scores=2
    textblob_score = float( item["textblob"])
    if textblob_score<0.2:
        textblob_score=1
    else :
        textblob_score=2
    lst_of_sia.append(sia_scores)
    lst_of_text.append(textblob_score)
array_of_sia=np.array(lst_of_sia)
array_of_text=np.array(lst_of_text)
# array_of_lists = np.array(lst_of_lists)

print(array_of_sia)
print(array_of_text)




    # console.print(f"[white on red]{df.iloc[i]['rate']}:[/] | [yellow]{res['sia']}[/] | [green]{res['textblob']}[/] {df.iloc[i]['all_text']}\n [yellow]{text}")

# print(lst)
# samples_df['all_text']=samples_df['all_text'].apply(pre_process)
# print(len(samples_df.iloc[0]['all_text']),len(samples_df.iloc[10]['all_text']))

# X_train, X_test, y_train, y_test = train_test_split(df.all_text, df.rate, test_size=0.2,train_size=9,stratify=df.rate)
# X_train=samples_df['all_text']
# y_train=samples_df['rate']
# X_test=samples_test['all_text']
y_test=samples_test['rate']

# vecotrizer=CountVectorizer()
# X_train_count = vecotrizer.fit_transform(X_train.values)
# X_test_count = vecotrizer.transform(X_test.values)
# model = GaussianNB()
# X_train_count=X_train_count.toarray()
# X_test_count=X_test_count.toarray()

# Train the model on the training data.
# model.fit(X_train_count, y_train)
# print(model.score(X_test_count,y_test))

# Make predictions on the test data.
y_pred_sia = array_of_sia
y_pred_text= array_of_text

# file_name = 'modelnltk.pkl'

# # Pickle the model.
# with open(file_name, 'wb') as f:
#     pickle.dump(model, f)

# # Close the file.
# f.close()

# Evaluate the model on the test data.
print("\n\n",classification_report(y_test, y_pred_sia))
print("\n\n",classification_report(y_test, y_pred_text))



# predictions=model.predict(X_test_count)
# cm = confusion_matrix(predictions, y_test, labels=model.classes_)
# disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
# disp.plot()
# plt.show()


# file_name = 'model.pkl'

# # Pickle the model.
# with open(file_name, 'wb') as f:
#     pickle.dump(model, f)

# # Close the file.
# f.close()







# print("\n\n\n")

# for i in range(df.shape[0]):
#     text = df.iloc[i]['summary']
#     text = pre_process(str(text))
#     res["sia"] = sia.polarity_scores(text)
#     res["textblob"] = sa_nlp(text)._.polarity

# #   filtered_txt=' '.join([word.text for word in doc if not(word._.polarity==0.0)])
#     # console.print(f"[white on red]{samples_df.iloc[i]['rate']}:[/] | [yellow]{res['sia']}[/] | [green]{res['textblob']}[/] {samples_df.iloc[i]['summary']}\n [yellow]{text}")


# df['summary']=df['summary'].apply(pre_process)
# # print(len(samples_df.iloc[0]['all_text']),len(samples_df.iloc[10]['all_text']))

# X_train, X_test, y_train, y_test = train_test_split(df.summary, df.rate, test_size=0.2,train_size=9,stratify=df.rate)

# vecotrizer=CountVectorizer()
# X_train_count = vecotrizer.fit_transform(X_train.values)
# X_test_count = vecotrizer.transform(X_test.values)
# model = GaussianNB()
# X_train_count=X_train_count.toarray()
# X_test_count=X_test_count.toarray()

# # Train the model on the training data.
# model.fit(X_train_count, y_train)
# print(model.score(X_test_count,y_test))

# # Make predictions on the test data.
# y_pred = model.predict(X_test_count)

# # Evaluate the model on the test data.
# print("\n\n\n",classification_report(y_test, y_pred))



# predictions=model.predict(X_test_count)
# cm = confusion_matrix(predictions, y_test, labels=model.classes_)
# disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
# disp.plot()
# plt.show()

# for i, row in df.iterrows():
#     text = row['sum_text']
#     text = str(text)
#     # myid = int(row['Id'])  
#     res[myid] = sia.polarity_scores(text)



# vaders = pd.DataFrame(res).T
# vaders.reset_index(inplace=True) 
# vaders.rename(columns={'index': 'Id'}, inplace=True)  
# vaders = vaders.merge(df, on='Id', how='left')


# x = vaders.drop('target_column_name', axis=1)  
# y = vaders['target_column_name']  
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)


# with open("x_train.pkl", "wb") as f:
#     pickle.dump(x_train, f)
# with open("y_train.pkl", "wb") as f:
#     pickle.dump(y_train, f)
# with open("x_test.pkl", "wb") as f:
#     pickle.dump(x_test, f)
# with open("y_test.pkl", "wb") as f:
#     pickle.dump(y_test, f)
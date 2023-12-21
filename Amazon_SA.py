# series = pd.Series(['A', 'B', 'C', 'D'])
# list_of_lists = [list(row) for row in zip(series)]


import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from rich.console import Console

from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacytextblob')


console = Console() 

console.print(f"[white on green]{nlp.pipe_names}")


df=pd.read_csv('train.csv', encoding="utf-8", header=0)
console.print(f"[white on green] Preparing Dataframe ...")

df['all_text'] = df['summary'] + ' ' + df['text']
# df.drop(columns=['summary','text'],inplace=True)

corpus = df['all_text'].to_list()

# print(corpus[:4])


doc=nlp(corpus)

# samples = df.groupby('rate').sample(15, random_state=123)

# console.print(f"[white on green] Preview Dataframe ...")

# preview_samples=samples.groupby('rate').sample(2)

# for i in range(preview_samples.shape[0]):
#     console.print(f"[white on red]{preview_samples.iloc[i]['rate']}:[/] {preview_samples.iloc[i]['all_text']}\n")
#     doc=nlp(preview_samples.iloc[i]['all_text'])
#     filtered_txt=' '.join([word.text for word in doc if not(word._.polarity==0.0)])
#     console.print(f"Pre-processed: [white on red]{nlp(filtered_txt)._.polarity}[/] [yellow]{filtered_txt}\n")

"""
# samples['rate'].value_counts()

==============================================================================
samples['y_pred'] = samples['all_text'].apply(lambda x: nlp(x)._.polarity)
samples['assessment'] = samples['all_text'].apply(lambda x: nlp(x)._.assessments)


for i in range(samples.shape[0]):
    console.print(samples.iloc[i][['all_text']])
    console.print(samples.iloc[i][['rate','y_pred', 'assessment']])
    console.print(samples.iloc[i][['assessment']])
    console.print('\n')

"""
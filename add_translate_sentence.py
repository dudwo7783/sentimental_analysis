from google.cloud import translate_v2 as translate
import os
import json
import pandas as pd

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/kim-youngjae/Desktop/Translation-34bjkdfg84.json"
translate_client = translate.Client()



with open("./data/friends_train.json", encoding='utf-8', mode='r') as json_file:
    json_train = json.load(json_file)
    sentences = []
    y = []
    for diag in json_train:
        for row in diag:
            sentences.append(row['utterance'])
            y.append(row['emotion'])

    df = pd.DataFrame({'Id': range(0, len(sentences)), 'Sentences': sentences, 'Label': y})

multi_df = df[df['Label'].isin(['anger', 'disgust', 'fear', 'joy', 'sadness'])].reset_index(drop=True)

translate_client = translate.Client()
multi_df['german'] = multi_df['Sentences'].apply(
    lambda x: translate_client.translate(x, target_language="de")["translatedText"])

multi_df.to_csv('./add_german.csv', index=False)

multi_df['italia'] = multi_df['Sentences'].apply(
    lambda x: translate_client.translate(x, target_language="it")["translatedText"])

multi_df.to_csv('./add_italia.csv', index=False)

multi_df['france'] = multi_df['Sentences'].apply(
    lambda x: translate_client.translate(x, target_language="fr")["translatedText"])

multi_df.to_csv('./add_france.csv', index=False)

df['german_to_en'] = df['german'].apply(
    lambda x: translate_client.translate(x, target_language="en")["translatedText"])
df.to_csv('./german_to_en.csv', index=False)

df['italia_to_en'] = df['italia'].apply(
    lambda x: translate_client.translate(x, target_language="en")["translatedText"])
df.to_csv('./italia_to_en.csv', index=False)

df['france_to_en'] = df['france'].apply(
    lambda x: translate_client.translate(x, target_language="en")["translatedText"])

df.to_csv('./france_to_en.csv', index=False)

df = pd.read_csv('./france_to_en.csv')

df1 = df[['german_to_en', 'Label']]
df1.columns = ['Sentences', 'Label']
df2 = df[['italia_to_en', 'Label']]
df2.columns = ['Sentences', 'Label']
df3 = df[['france_to_en', 'Label']]
df3.columns = ['Sentences', 'Label']

additional_df = pd.concat([df1, df2, df3])
additional_df.to_csv('./additional_friends.csv', index=False)
class_size = additional_df.groupby('Label').size()
print(pd.DataFrame({'class_size' :class_size}).T)

#!pip install transformers
# !pip install sentencepiece

import tensorflow as tf
import torch

from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import random
import time
import datetime
from tqdm.notebook import tqdm

import json
import re

# GPU 디바이스 이름 구함
device_name = tf.test.gpu_device_name()

# GPU 디바이스 이름 검사
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')

# 디바이스 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


class FriendsDataset(Dataset):

    def __init__(self, filename, mode='train', concat_size=3):
        # 일부 값중에 NaN이 있음...

        self.concat_size = concat_size
        self.mode = mode
        self.maxlen = 82
        dataset = self.read_data(filename)

        if self.mode == 'train':
            # dataset = dataset.groupby('Label', group_keys=False).apply(lambda x: x.sample(int(len(x)*3), replace=True) if len(x) < 1000 else x.sample(int(len(x)*0.9), replace=False) if len(x) < 2200 else x.sample(int(len(x)*0.5), replace=False)).reset_index(drop=True)
            dataset = dataset.dropna(axis=0)
            dataset.drop_duplicates(subset=['Sentences'], inplace=True)

            # dataset = dataset.drop(([dataset['Label'].isin([0,1,6])) & (dataset['Sentences'].apply(lambda x : len(x.split() <3))).index)
        self.dataset = dataset
        # 중복제거

        self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator", truncation=True,
                                                          max_length=self.maxlen,
                                                          pad_to_max_length=True,
                                                          add_special_tokens=False, do_lower_case=True)

        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if (self.mode == 'train') or (self.mode == 'json_test'):
            row = self.dataset.iloc[idx, 1:3].values
            text = row[0]
            y = row[1]

            tokenized_texts = self.tokenizer.tokenize(text)
            input_ids = [[self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]]
            input_ids = pad_sequences(input_ids, maxlen=self.maxlen, dtype="long", truncating="post", padding="post")

            attention_mask = []
            for seq in input_ids:
                seq_mask = [int(i > 0) for i in seq]
                attention_mask.append(seq_mask)

            return torch.tensor(input_ids[0]), torch.tensor(attention_mask[0]), y

        else:
            row = self.dataset.iloc[idx, 1:2].values
            text = row[0]

            tokenized_texts = self.tokenizer.tokenize(text)
            input_ids = [[self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]]
            input_ids = pad_sequences(input_ids, maxlen=self.maxlen, dtype="long", truncating="post", padding="post")

            attention_mask = []
            for seq in input_ids:
                seq_mask = [int(i > 0) for i in seq]
                attention_mask.append(seq_mask)

            return torch.tensor(input_ids[0]), torch.tensor(attention_mask[0])

    def read_data(self, filename):
        def cleaning(sentence):
            contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                                   "could've": "could have", "couldn't": "could not",
                                   "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                                   "hasn't": "has not", "haven't": "have not",
                                   "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                                   "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                                   "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                                   "i've": "i have", "isn't": "is not", "it'd": "it would",
                                   "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                                   "it's": "it is", "let's": "let us", "ma'am": "madam",
                                   "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                                   "mightn't've": "might not have",
                                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                                   "needn't": "need not", "needn't've": "need not have",
                                   "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                                   "shan't": "shall not", "sha'n't": "shall not",
                                   "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                                   "she'll": "she will", "she'll've": "she will have",
                                   "she's": "she is", "should've": "should have", "shouldn't": "should not",
                                   "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                                   "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                                   "that's": "that is", "there'd": "there would", "there'd've": "there would have",
                                   "there's": "there is", "here's": "here is", "they'd": "they would",
                                   "they'd've": "they would have", "they'll": "they will",
                                   "they'll've": "they will have",
                                   "they're": "they are", "they've": "they have", "to've": "to have",
                                   "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                                   "we'll": "we will",
                                   "we'll've": "we will have", "we're": "we are", "we've": "we have",
                                   "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                                   "what're":
                                       "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                                   "when've": "when have", "where'd": "where did", "where's": "where is",
                                   "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                                   "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                                   "will've": "will have", "won't": "will not", "won't've": "will not have",
                                   "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                                   "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                                   "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                                   "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                                   "you're": "you are", "you've": "you have",
                                   "gonna": "going to", "wanna": "want to", "gotta": "got to", "ya": "you"}
            specials = ["’", "‘", "´", "`"]

            sentence = sentence.replace("&#39;", "'").replace("\x91", "'").replace("\x92", "'").lower()
            for s in specials:
                sentence = sentence.replace(s, "'")
            sentence = ' '.join(
                [contraction_mapping[t] if t in contraction_mapping else t for t in sentence.split(" ")]).replace("'s",
                                                                                                                  "")
            sentence = re.sub(r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', r'', sentence)
            # sentence = ' '.join([t for t in sentence.split(" ") if not t in stopwords]).replace("'s", "")

            return sentence

        def labeltoint(str):
            return {'non-neutral': 0,
                    'neutral': 1,
                    'joy': 2,
                    'sadness': 3,
                    'fear': 4,
                    'anger': 5,
                    'surprise': 6,
                    'disgust': 7}[str]

        train_data = []

        if (self.mode == 'train') or (self.mode == 'json_test'):
            with open(filename, encoding='utf-8', mode='r') as json_file:
                json_train = json.load(json_file)

            if self.mode == 'train':
                additional_sentences = pd.read_csv("./data/additional_friends.csv", encoding='utf-8')

                for s, l in zip(additional_sentences['Sentences'], additional_sentences['Label']):
                    c_s = cleaning(s)
                    train_data.append([f"[CLS] {c_s} [SEP]", l])

            for diag in json_train:
                cur_diag = []
                for row in diag:
                    cur_diag.append([cleaning(row['utterance']), row['emotion']])
                for i in range(self.concat_size - 1):
                    cur_diag.insert(0, [None, ''])
                for i in range(self.concat_size, len(cur_diag) + 1):
                    train_data.append(["[CLS] " + "".join(
                        [cur_diag[j][0] + ' [SEP]' for j in range(i - self.concat_size, i) if
                         cur_diag[j][0] is not None]), cur_diag[i - 1][1]])

            sentences = [td[0] for td in train_data]
            y = [labeltoint(td[1]) for td in train_data]

            return pd.DataFrame({'Id': range(0, len(sentences)), 'Sentences': sentences, 'Label': y})

        else:
            df = pd.read_csv(filename, encoding='unicode_escape')
            dialog_num = df['i_dialog'].unique().tolist()
            dialog_num.sort()

            for diag in dialog_num:
                cur_diag = []
                for row in df[df['i_dialog'] == diag]['utterance'].values:
                    cur_diag.append(cleaning(row))

                for i in range(self.concat_size - 1):
                    cur_diag.insert(0, None)

                for i in range(self.concat_size, len(cur_diag) + 1):
                    train_data.append("[CLS] " + "".join(
                        [cur_diag[j] + ' [SEP]' for j in range(i - self.concat_size, i) if cur_diag[j] is not None]))

            return pd.DataFrame({'Id': range(0, len(train_data)), 'Sentences': train_data})


model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=8).to(device)


# 정확도 계산 함수
def flat_accuracy(preds, labels, confusion_matrix):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for t, p in zip(labels_flat, pred_flat):
        confusion_matrix[t, p] += 1

    return np.sum(pred_flat == labels_flat) / len(labels_flat), confusion_matrix


# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
nb_classes = 8

# 그래디언트 초기화
model.zero_grad()

# 에폭만큼 반복
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # 학습률
                  eps=1e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                  )


def train():
    # 옵티마이저 설정

    # 에폭수
    epochs = 7

    # 총 훈련 스텝 : 배치반복 횟수 * 에폭
    batch_size = 24
    train_dataset = FriendsDataset("./data/friends_train.json", 'train', concat_size=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_steps = len(train_loader) * epochs

    # 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # 시작 시간 설정
        t0 = time.time()

        # 로스 초기화
        total_loss = 0
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        confusion_matrix = torch.zeros(nb_classes, nb_classes)

        # 훈련모드로 변경
        model.train()

        # 데이터로더에서 배치만큼 반복하여 가져옴
        for step, batch in enumerate(train_loader):
            # 경과 정보 표시
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            # Forward 수행
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # 로스 구함
            loss, logits = outputs[0], outputs[1]

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            tmp_eval_accuracy, confusion_matrix = flat_accuracy(logits, label_ids, confusion_matrix)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

            # loss = outputs[0]

            # 총 로스 계산
            total_loss += loss.item()

            # Backward 수행으로 그래디언트 계산
            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 그래디언트를 통해 가중치 파라미터 업데이트
            optimizer.step()

            # 스케줄러로 학습률 감소
            scheduler.step()

            # 그래디언트 초기화
            model.zero_grad()

            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print(" Class Accuracy: {}")
            print(confusion_matrix.diag() / confusion_matrix.sum(1))

        # 평균 로스 계산
        avg_train_loss = total_loss / len(train_loader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        if eval_accuracy / nb_eval_steps >= 0.6:
            break

    print("")
    print("Training complete!")


def test():
    def inttolabel(label):
        return {0: 'non-neutral',
                1: 'neutral',
                2: 'joy',
                3: 'sadness',
                4: 'fear',
                5: 'anger',
                6: 'surprise',
                7: 'disgust'}[label]

    test_dataset = FriendsDataset("./data/friends_test.json", 'json_test', concat_size=1)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    eval_accuracy = 0
    nb_eval_steps = 0
    pred_y = []
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for batch in tqdm(test_loader):
            optimizer.zero_grad()
            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # CPU로 데이터 이동
            predicted = torch.argmax(outputs[1], dim=1)
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_y.append(predicted)

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            tmp_eval_accuracy, confusion_matrix = flat_accuracy(logits, label_ids, confusion_matrix)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print(" Class Accuracy: ")
            print(confusion_matrix.diag() / confusion_matrix.sum(1))

    total_pred = torch.cat(pred_y, dim=0).cpu().numpy()
    x_df = pd.DataFrame({'Id': range(0, len(total_pred)), 'Predicted': total_pred})
    x_df['Predicted'] = x_df['Predicted'].apply(lambda x: inttolabel(x))

    x_df.to_csv('./sample.csv', index=False)


def leaderboard_test():
    def inttolabel(label):
        return {0: 'non-neutral',
                1: 'neutral',
                2: 'joy',
                3: 'sadness',
                4: 'fear',
                5: 'anger',
                6: 'surprise',
                7: 'disgust'}[label]

    test_dataset = FriendsDataset("./data/Friends/en_data.csv", "test", 1)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    pred_y = []
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in tqdm(test_loader):
            optimizer.zero_grad()
            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)
            # 배치에서 데이터 추출
            b_input_ids, b_input_mask = batch

            # 배치에서 데이터 추출
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)[0]

            # CPU로 데이터 이동
            predicted = torch.argmax(outputs, dim=1)
            logits = outputs[1].detach().cpu().numpy()

            pred_y.append(predicted)

    total_pred = torch.cat(pred_y, dim=0).cpu().numpy()
    x_df = pd.DataFrame({'Id': range(0, len(total_pred)), 'Predicted': total_pred})
    x_df['Predicted'] = x_df['Predicted'].apply(lambda x: inttolabel(x))

    x_df.to_csv('./sample123.csv', index=False)


if __name__ == "__main__":
    # train()
    # test()
    leaderboard_test()

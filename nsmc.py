#!pip install transformers

import tensorflow as tf
import torch

from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import time
import datetime
from tqdm.notebook import tqdm

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


class NSMCDataset(Dataset):

    def __init__(self, csv_file, mode='train'):
        # 일부 값중에 NaN이 있음...
        if mode == 'train':
            sep = '\t'
            encoding = 'UTF8'
        else:
            sep = ','
            encoding = 'CP949'

        self.dataset = pd.read_csv(csv_file, sep=sep, encoding=encoding).dropna(axis=0)
        # 중복제거
        # self.dataset.drop_duplicates(subset=['document'], inplace=True)
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.mode = mode

        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.mode == 'train':
            row = self.dataset.iloc[idx, 1:3].values
            text = row[0]
            y = row[1]

            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=256,
                pad_to_max_length=True,
                add_special_tokens=True
            )

            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            return input_ids, attention_mask, y
        else:
            row = self.dataset.iloc[idx, 1:2].values
            text = row[0]

            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=256,
                pad_to_max_length=True,
                add_special_tokens=True
            )

            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            return input_ids, attention_mask


model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator").to(device)

# 옵티마이저 설정
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # 학습률
                  eps=1e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                  )

nb_classes = 2


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


def train():
    # 에폭수
    epochs = 7

    # 총 훈련 스텝 : 배치반복 횟수 * 에폭
    batch_size = 32
    train_dataset = NSMCDataset("./data/ratings_train.txt", 'train')
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

        if eval_accuracy / nb_eval_steps >= 9.0:
            break

    print("")
    print("Training complete!")


def test():
    nb_classes = 2
    model.eval()
    eval_accuracy = 0
    nb_eval_steps = 0
    pred_y = []
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    test_dataset = NSMCDataset("./data/ratings_train.txt", 'train')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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


def leaderboard_test():
    test_dataset = NSMCDataset("./data/data.csv", 'test')
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

    x_df.to_csv('./sample123.csv', index=False)


if __name__ == "__main__":
    train()
    test()
    leaderboard_test()
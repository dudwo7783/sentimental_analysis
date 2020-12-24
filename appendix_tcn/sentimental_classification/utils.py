import torch
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from transformers import ElectraModel, ElectraTokenizer, ElectraForSequenceClassification
from sklearn.model_selection import train_test_split

from torch.utils.data import  TensorDataset, DataLoader

def tokenize(data):
    sentences = data['document']
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

    labels = data['label'].values

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=128, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [int(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, labels, attention_masks


def data_generator(root, batch_size):
    train = pd.read_csv("./data/nsmc/ratings_train.txt", sep='\t').dropna()
    test = pd.read_csv("./data/nsmc/ratings_test.txt", sep='\t').dropna()

    train_input_ids, train_labels, train_attention_masks = tokenize(train)
    test_input_ids, test_labels, test_attention_masks = tokenize(test)

    # 훈련셋과 검증셋으로 분리
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_input_ids,
                                                                                        train_labels,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)

    # 어텐션 마스크를 훈련셋과 검증셋으로 분리
    train_masks, validation_masks, _, _ = train_test_split(train_attention_masks,
                                                           train_input_ids,
                                                           random_state=2018,
                                                           test_size=0.1)

    # 데이터를 파이토치의 텐서로 변환
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    test_inputs = torch.tensor(test_input_ids)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_attention_masks)
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    train_set = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_labels), torch.tensor(train_masks))
    test_set = TensorDataset(torch.tensor(test_inputs), torch.tensor(test_labels), torch.tensor(test_masks))
    validation_set = TensorDataset(torch.tensor(validation_inputs), torch.tensor(validation_labels), torch.tensor(validation_masks))


    train_loader = DataLoader(train_set, batch_size=batch_size)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader, validation_loader

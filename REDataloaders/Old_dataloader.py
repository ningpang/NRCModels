import torch
# import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
import json

class data_sampler(object):
    def __init__(self, config=None, seed=None):
        self.config = config
        self.additional_special_tokens=['[E11]', '[E12]', '[E21]', '[E22]']
        self.id2rel, self.rel2id = self.read_relation(config.relation_file)
        self.word2id, self.word_vec = self.get_word_vec(config.word_vec_file)
        # random sample
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)

        # generate data
        self.training_data, self.valid_data, self.test_data = self.read_data(config.data_file)

    def get_word_vec(self, file):
        if not os.path.exists(file):
            print("[Error] Data file does not exists !")
            assert 0
        word_vec = json.load(open(file, 'r'))
        word_total = len(word_vec)
        word_dim = len(word_vec[0]['vec'])
        word_vec_mat = np.zeros((word_total, word_dim), dtype=np.float32)
        word2id = {}
        for cur_id, word in enumerate(word_vec):
            w = word['word'].lower()
            word2id[w] = cur_id
            word_vec_mat[cur_id, :] = word['vec']
            word_vec_mat[cur_id] = word_vec_mat[cur_id] / np.sqrt((np.sum(word_vec_mat[cur_id] ** 2)))

        UNK = word_total
        PAD = word_total + 1
        word2id['UNK'] = UNK
        word2id['PAD'] = PAD
        return word2id, word_vec_mat

    def read_data(self, file):
        data = json.load(open(file, 'r'), encoding='utf-8')
        train_dataset = [[] for i in range(self.config.num_of_relation)]
        valid_dataset = [[] for i in range(self.config.num_of_relation)]
        test_dataset = [[] for i in range(self.config.num_of_relation)]

        # t = self.tokenizer.convert_ids_to_tokens(torch.tensor([102]))

        for relid in range(self.config.num_of_relation):
            relation = self.id2rel[relid]
            rel_samples = data[relation]
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):
                tokenized_sample = {}
                tokenized_sample['relation'] = self.rel2id[sample['relation']]
                tokenized_sample['tokens'], tokenized_sample['pos1'], tokenized_sample['pos2'],\
                tokenized_sample['length'], tokenized_sample['mask'] \
                    = self.tokenizer(sample['tokens'], max_length=self.config.max_length)

                if self.config.task_name == 'FewRel':
                    if i < self.config.num_of_train:
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                    elif i < self.config.num_of_train + self.config.num_of_val:
                        valid_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                elif self.config.task_name == 'tacred':
                    if i < len(rel_samples) // 5 and count <= 40:
                        count += 1
                        valid_dataset[self.rel2id[relation]].append(tokenized_sample)
                        test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        if count1 >= 320:
                            break
                else:
                    raise Exception('Datasets only include tarced and fewrel')

        return train_dataset, valid_dataset, test_dataset

    def read_relation(self, file):
        id2rel = json.load(open(file, 'r'), encoding='utf-8')
        rel2id = {}
        for i, rel in enumerate(id2rel):
            rel2id[rel] = i
        return id2rel, rel2id

    def tokenizer(self, sentence, max_length=256):
        tokens = np.zeros(max_length, dtype=np.int32)
        pos1 = np.zeros(max_length, dtype=np.int32)
        pos2 = np.zeros(max_length, dtype=np.int32)
        mask = np.zeros(max_length, dtype=np.int32)
        # preprocess
        tokens_wo_marker = []
        head, tail = 0, 0
        for i, token in enumerate(sentence):
            if token in self.additional_special_tokens:
                if token == '[E11]':
                    head = i
                if token == '[E21]':
                    tail = i
            else:
                tokens_wo_marker.append(token)
        if tail > head:
            tail -= 2
        else:
            head -= 2
        # token
        for i, token in enumerate(tokens_wo_marker):
            if i < max_length:
                if token in self.word2id:
                    tokens[i] = self.word2id[token]
                else:
                    tokens[i] = self.word2id['UNK']
        for i in range(i+1, max_length):
            tokens[i] = self.word2id['PAD']
        # length
        length = len(tokens_wo_marker)
        if len(tokens_wo_marker) > max_length:
            length = max_length
        # position
        if head >= max_length:
            head = max_length-1
        if tail >= max_length:
            tail = max_length-1
        pos_min = min(head, tail)
        pos_max = max(head, tail)
        # mask
        for i in range(max_length):
            pos1[i] = i - head + max_length
            pos2[i] = i - tail + max_length
            if i >= length:
                mask[i] = 0
            elif i <= pos_min:
                mask[i] = 1
            elif i <= pos_max:
                mask[i] = 1
            else:
                mask[i] = 1

        return tokens, pos1, pos2, length, mask


class data_set(Dataset):
    def __init__(self, data, config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        label = torch.tensor([item['relation'] for item in data])
        length = torch.tensor([item['length'] for item in data])
        tokens = [torch.tensor(item['tokens']) for item in data]
        pos1 = [torch.tensor(item['pos1']) for item in data]
        pos2 = [torch.tensor(item['pos2']) for item in data]
        mask = [torch.tensor(item['mask']) for item in data]
        return (label, length, tokens, pos1, pos2, mask)

def get_data_loader(config, data, shuffle=True, drop_last = False, batch_size=None):
    dataset = data_set(data, config)
    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last
    )
    return data_loader

def get_old_loaders(config):
    sampler = data_sampler(config)
    training, valid, test = sampler.training_data, sampler.valid_data, sampler.test_data
    traindata, validdata, testdata = [], [], []
    for i in range(config.num_of_relation):
        traindata += training[i]
        validdata += valid[i]
        testdata += test[i]

    trainloader = get_data_loader(config, traindata)
    validloader = get_data_loader(config, validdata, batch_size=1)
    testloader = get_data_loader(config, testdata, batch_size=1)

    word2id, word_vec = sampler.word2id, sampler.word_vec

    return trainloader, validloader, testloader, word2id, word_vec
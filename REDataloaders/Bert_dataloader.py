import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import json
import random
from transformers import BertTokenizer

class data_sampler(object):
    def __init__(self, config=None, seed=None):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, additional_special_tokens=['[E11]', '[E12]', '[E21]', '[E22]'])
        self.id2rel, self.rel2id = self.read_relation(config.relation_file)

        # random sample
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)

        # generate data
        self.training_data, self.valid_data, self.test_data = self.read_data(config.data_file)

    # def get_data(self):
    #     training_data = {}
    #     valid_data = {}
    #     test_data = {}
    #     for i in range(self.config.num_of_relation):
    #         training_data[self.id2rel[i]] = self.training_data[i]
    #         valid_data[self.id2rel[i]] = self.valid_data[i]
    #         test_data[self.id2rel[i]] = self.test_data[i]
    #
    #     return training_data, valid_data, test_data


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
                tokenized_sample['tokens'] = self.tokenizer.encode(' '.join(sample['tokens']),
                                                                   padding='max_length',
                                                                   truncation=True,
                                                                   max_length=self.config.max_length)
                length = np.argwhere(np.array(tokenized_sample['tokens'])==102)[0][0]
                mask = [1]*length + [0]*(self.config.max_length-length)

                tokenized_sample['length'] = length
                tokenized_sample['mask'] = mask
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
        masks = [torch.tensor(item['mask']) for item in data]
        return (label, length, tokens, masks)

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

def get_bert_loaders(config):
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
    return trainloader, validloader, testloader
#!/usr/bin/env python
# coding: utf-8

import json
from PIL import Image
import os
import pickle
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
from matplotlib.pyplot import imshow
import pickle
import argparse
from collections import Counter
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class Im2PDataSet(data.Dataset):
    def __init__(self,
                 image_dir,
                 caption_json,
                 data_json,
                 vocabulary,
                 transform=None,
                 s_max=6,
                 n_max=50):
        self.image_dir = image_dir
        self.caption = JsonReader(caption_json)
        self.data = JsonReader(data_json)
        self.vocabulary = vocabulary
        self.transform = transform
        self.s_max = 6
        self.n_max = 50
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    
    def __getitem__(self, index):
        """

        :param index: id
        :return:
            image: (3, 224, 224)
            target: (sentence_num, word_num)
            image_id: 1
        """
        image_id = str(self.data[index])
        paragraph = self.caption[image_id]['paragraph']

        image = Image.open(os.path.join(self.image_dir, "{}.jpg".format(image_id))).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        target = list()
        word_num = 0
        for i, sentence in enumerate(paragraph.split('. ')):
            if i >= self.s_max:
                break
            sentence = sentence.lower().replace('.', '').replace(',', '').split()
            if len(sentence) == 0 or len(sentence) > self.n_max:
                continue
            tokens = list()
            tokens.append(self.vocabulary('<start>'))
            tokens.extend([self.vocabulary(token) for token in sentence])
            tokens.append(self.vocabulary('<end>'))
            if word_num < len(tokens):
                word_num = len(tokens)
            target.append(tokens)
        sentence_num = len(target)
        return image, image_id, target, sentence_num, word_num
    
    def __len__(self):
        return len(self.data)
    
def collate_fn(data):
    """
    :param data: list of tuple (image, target, image_id)
    :return:
        images: (batch_size, 3, 224, 224)
        targets: (batch_size, 6, 50)
        lengths: (batch_size, s_max)
        image_id: (batch_size, )
    """
    images, image_id, captions, sentences_num, words_num = zip(*data)
    images = torch.stack(images, 0)

    max_sentence_num = max(sentences_num)
    max_word_num = max(words_num)

    targets = np.zeros((len(captions), max_sentence_num, max_word_num))
    prob = np.zeros((len(captions), max_sentence_num))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i, j] = 1

    targets = torch.Tensor(targets).long()
    prob = torch.Tensor(prob)

    return images, image_id, targets, prob

def get_loader(image_dir,
           caption_json,
           data_json,
           vocabulary,
           transform,
           batch_size,
           shuffle):
    dataset = Im2PDataSet(image_dir=image_dir,
                      caption_json=caption_json,
                      data_json=data_json,
                      vocabulary=vocabulary,
                      transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader

class JsonReader(object):
    def __init__(self, json_file):
        self.data = self.__read_json(json_file)

    def __read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<end>')
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __getitem__(self, item):
        if item > len(self.id2word):
            return '<unk>'
        return self.id2word[item]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_file, threshold):
    caption_reader = JsonReader(json_file)
    counter = Counter()

    for items in caption_reader:
        paragraph = items['paragraph']
        paragraph = paragraph.replace(',', ' ').replace('.', '').replace('\"', '')
        counter.update(paragraph.lower().split(' '))
    words = [word for word, cnt in counter.items() if cnt >= threshold and word != '']
    vocab = Vocabulary()

    for _, word in enumerate(words):
        if len(word) > 1:
            vocab.add_word(word)
    return vocab





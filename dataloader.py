import torch
from torch.utils.data import Dataset,DataLoader
import csv
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys

csv.field_size_limit(sys.maxsize)


class AGNEWS_Dataset(Dataset):
    """
    Defines the AG News dataset
    """

    def __init__(self,csv_path,alphabet_path,max_seq_len):
        """
        Initializes the dataset
        @params csv_path (str): Path to csv file that contains train/test data
        @params alphabet_path (str): Path to json file that contains the
        characters considered
        @params max_seq_len (int): Maximum number of characters considered]
        for input
        """
        self.max_seq_len = max_seq_len
        with open(alphabet_path) as f:
            self.alphabet = json.load(f)
        with open(csv_path) as f:
            self.data = csv.reader(f,delimiter=',')
            self.data = list(self.data)

    def __getitem__(self,idx):
        """
        Returns a text, class pair
        @params idx (int): Index into the dataset
        @returns (self.seq,self.cls) tuple(torch.Tensor,int): Returns a tensor
        of shape (num_characters,max_seq_len) representing the input text and an
        integer representing the class index
        """
        self.cls = int(self.data[idx][0])
        self.seq = torch.zeros(len(self.alphabet),self.max_seq_len)
        seq_len = 0
        sequence = "".join(self.data[idx][1:])
        sequence = sequence[::-1]
        for char in sequence:
            if seq_len > self.max_seq_len:
                break
            try:
                self.seq[self.alphabet.index(char)][seq_len] = 1
            except:
                pass
            seq_len += 1
        return self.seq,self.cls

    def __len__(self):
        """
        Returns the dataset length
        """
        return len(list(self.data))

def one_hot(data,alphabet):
    """
    Converts a character to its one-hot vector representation
    @params data (char): The character that is input to the CNN
    @params alphabet (list): The list of characters considered.
    NOTE: Characters outside the alphabet are considered to be a zero vector
    @returns t (torch.Tensor): Tensor of shape (len(alphabet))
    """
    t = torch.zeros(len(alphabet))
    try:
        t[alphabet.index(data)] = 1
    except:
        return t
    return t

import jsonlines
import numpy as np
from torch.utils.data import Dataset


class CausalLMDataset(Dataset):

    def __init__(self, jsonfile: str, train: bool=True):
        self.texts = CausalLMDataset.load_dataset(jsonfile, train)

    @staticmethod
    def load_dataset(jsonfile: str, train: bool=True):
        texts = []
        with jsonlines.open(jsonfile, mode="r") as f:
            for line in f:
                if train:
                    texts.append("// " + line["comment"] + " \n" + line["function"] + "\n\n\n")
                else: # inference
                    texts.append("// " + line["comment"] + " \n" + line["signature"] + " {")
        return texts

    def __getitem__(self, idx):
        return self.texts[idx]

    def __len__(self):
        return len(self.texts)

    def examples(self, n=5):
        idxs = np.random.choice(list(range(len(self.texts))), size=n, replace=False)
        return [self.texts[idx] for idx in idxs]



class Seq2SeqLMDataset(Dataset):

    def __init__(self, jsonfile: str):
        self.queries, self.answers = Seq2SeqLMDataset.load_dataset(jsonfile)

    @staticmethod
    def load_dataset(jsonfile: str):
        queries = []
        answers = []
        with jsonlines.open(jsonfile, mode="r") as f:
            for line in f:
                queries.append("// " + line["comment"] + " " + line["signature"])
                answers.append(line["body"] + "\n\n\n")
        return queries, answers

    def __getitem__(self, idx):
        return self.queries[idx], self.answers[idx]

    def __len__(self):
        return len(self.queries)

    def examples(self, n=5):
        idxs = np.random.choice(list(range(len(self.texts))), size=n, replace=False)
        return {
            "queries": [self.queries[idx] for idx in idxs],
            "answers": [self.answers[idx] for idx in idxs]
        }

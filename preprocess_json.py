import csv
import jsonlines
import numpy as np

raw_function_train = "./sc-data/train_function_clean.csv"
raw_comment_train = "./sc-data/train_comment_clean.csv"
raw_function_test = "./sc-data/test_function_clean.csv"
raw_comment_test = "./sc-data/test_comment_clean.csv"

train_json = "./data-json/train.jsonl"
test_json = "./data-json/test.jsonl"


def preprocess(function_file, comment_file, json_file):
    datas = []
    lengths = []
    with open(comment_file, newline="") as f:
        rows = csv.reader(f)
        for row in rows:
            row = row[0].strip()
            datas.append({"comment": row})
    with open(function_file, newline="") as f:
        rows = csv.reader(f)
        for idx, row in enumerate(rows):
            row = row[0].strip("\"")
            body_begin_idx = row.find('{')
            signature = row[:body_begin_idx]
            body = row[body_begin_idx:]
            datas[idx]["function"] = row.strip()
            datas[idx]["signature"] = signature.strip()
            datas[idx]["body"] = body.lstrip("{").rstrip("}").strip()
            lengths.append(len(datas[idx]["comment"]) + len(datas[idx]["function"]))
    with jsonlines.open(json_file, mode='w') as f:
        f.write_all(datas)
    print("max len: ", max(lengths))
    print("mean len: ", np.mean(lengths))
    print("std len: ", np.std(lengths))

if __name__ == "__main__":
    preprocess(raw_function_train, raw_comment_train, train_json)
    preprocess(raw_function_test, raw_comment_test, test_json)

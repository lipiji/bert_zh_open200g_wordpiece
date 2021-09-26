import json

label_to_id = {}

def load_file(fpath):
    docs = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fs = json.loads(line)
            s1 = fs["sentence"]
            label = fs["label"]
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)
            docs.append((s1, label_to_id[label]))
    return docs

def write_file(docs, fpath):
    with open(fpath, "w") as f:
        for s1, label in docs:
            f.write(s1 +  "\t" + str(label) + "\n")

train_xs = load_file("/mnt/data/pjli/data/bert/tnews/train.json")
dev_xs = load_file("/mnt/data/pjli/data/bert/tnews/dev.json")
#test_xs = load_file("/mnt/data/pjli/data/bert/tnews/test.json")


print(len(train_xs), len(dev_xs))
write_file(train_xs, "./train.txt")
write_file(dev_xs, "./dev.txt")

with open("./labels.txt", "w") as f:
    for name, lid in label_to_id.items():
        f.write(name + "\t" + str(lid) + "\n")

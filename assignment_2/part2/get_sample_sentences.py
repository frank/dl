import torch


def load_sentences():
    checkpoint = torch.load('dumas_model.tar')
    sntcs = checkpoint['sentences']
    print("Loaded Checkpoint")
    return sntcs


sentences = load_sentences()
steps = list(sentences.keys())

for step in steps:
    print(step)
    print()
    for t in sentences[step]:
        print("[", t, "] :", sentences[step][t][0])
        print("[", t, "] :", sentences[step][t][1])
    print()


import torch
from nltk.tokenize import word_tokenize
from dataset import LabelIndexMap

model_to_load = "val_0.0627.pt"
vocab_path = 'jigsaw-toxic-comment-classification-challenge/vocab.txt'
device = torch.device('cuda')
threshold = 0.5

if __name__ == '__main__':

    classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # load model
    model = torch.load(model_to_load)
    model.eval()
    model.to(device)

    # load vocabulary
    vocab = LabelIndexMap.load(vocab_path)

    done = False
    while not done:

        input_text = input("Insert text:")
        tokens = list(map(lambda tok: vocab.label_to_index.get(tok.lower(), vocab['<UNK>']), word_tokenize(input_text)))
        input_tensor = torch.tensor(tokens, dtype=torch.long).to(device).unsqueeze(0)
        output_tensor = model(input_tensor).detach().cpu().squeeze().numpy()
        y_pred = (output_tensor > threshold)
        print("Predictions:")
        for class_label, class_predicted in zip(classes, y_pred):
            print("{}: {}".format(class_label, "yes" if class_predicted else "no"))
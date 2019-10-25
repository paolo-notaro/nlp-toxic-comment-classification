import torch
from numpy import array
from nltk.tokenize import word_tokenize
from dataset import LabelIndexMap

model_to_load = "loss=0.2414_f1=0.5478.pt"
vocab_path = 'jigsaw-toxic-comment-classification-challenge/vocab.txt'
device = torch.device('cpu')
thresholds = array([0.8, 0.98, 0.9, 0.99, 0.91, 0.97])

if __name__ == '__main__':

    classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    # load model
    model = torch.load(model_to_load)
    model.device = device
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
        y_pred = (output_tensor > thresholds)
        print("Predictions:")
        for class_label, prob, class_predicted in zip(classes, output_tensor, y_pred):
            print("{}: {} ({:.2f})".format(class_label, "yes" if class_predicted else "no", prob))

import torch
import json
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
class MultitaskABSAConcat(nn.Module):
    def __init__(self, num_aspects=10, num_polarities=4):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2", output_hidden_states=True)
        self.dropout = nn.Dropout(0.5)
        self.aspect_classifiers = nn.ModuleList([
            nn.Linear(self.phobert.config.hidden_size * 4, num_polarities) for _ in range(num_aspects)
        ])
        self.num_aspects = num_aspects
        self.num_polarities = num_polarities

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask,  output_hidden_states=True)
        concat_hidden = torch.cat(outputs.hidden_states[-4:], dim=-1)
        cls_output = concat_hidden[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        aspect_logits = [classifier(cls_output) for classifier in self.aspect_classifiers]
        logits = torch.cat(aspect_logits, dim=1)
        logits = logits.view(-1, self.num_aspects, self.num_polarities)
        return logits

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('vinai/phobert-base-v2', output_hidden_states=True)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size * 4, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        concat_hidden = torch.cat(tuple(hidden_states[-i] for i in range(1, 5)), dim=-1)
        cls_output = concat_hidden[:, 0, :]
        output = self.drop(cls_output)
        logits = self.out(output)
        return logits

# Load models and tokenizer
modelabsa_path = r"F:\slide\datn\UIT-ViSFD\ABSA\Concat.pth"
absa_model = MultitaskABSAConcat()
absa_model.load_state_dict(torch.load(modelabsa_path, map_location=device))
absa_model.to(device)

modelsa_path = r"F:\slide\datn\UIT-ViSFD\ABSA\SA.pth"
sa_model = SentimentClassifier(n_classes=3).to(device)
sa_model.load_state_dict(torch.load(modelsa_path, map_location=device))

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# Prediction functions
def predictABSA(model, tokenizer, device, comments):
    if isinstance(comments, str):
        comments = [comments]

    model.eval()
    inputs = tokenizer(comments, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        outputs = torch.softmax(outputs, dim=2)
        predictions = torch.argmax(outputs, dim=2)
        predictions = predictions.cpu().numpy()

    sentiment_to_index = {'Positive': 1, 'Negative': 0, 'Neutral': 2, 'None': 3}
    index_to_sentiment = {v: k for k, v in sentiment_to_index.items()}
    aspect_to_index = {'GENERAL': 0, 'SER&ACC': 1, 'SCREEN': 2, 'CAMERA': 3, 'FEATURES': 4,
                       'BATTERY': 5, 'PERFORMANCE': 6, 'STORAGE': 7, 'DESIGN': 8, 'PRICE': 9}
    index_to_aspect = {v: k for k, v in aspect_to_index.items()}

    predicted_aspects = []
    for comment_predictions in predictions:
        comment_aspects = []
        for aspect_idx, sentiment_idx in enumerate(comment_predictions):
            aspect = index_to_aspect[aspect_idx]
            sentiment = index_to_sentiment[sentiment_idx]
            comment_aspects.append((aspect, sentiment))
        predicted_aspects.append(comment_aspects)

    return predicted_aspects

def predictSA(model, tokenizer, device, comments):
    if isinstance(comments, str):
        comments = [comments]

    model.eval()
    inputs = tokenizer(comments, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        predictions = predictions.cpu().numpy()

    index_to_sentiment = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    predicted_sentiments = [index_to_sentiment[idx] for idx in predictions]

    return predicted_sentiments

# Load your JSON data
with open(r'F:\code\datn\webjs\data\test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process comments
for product in data:
    for comment in product.get('comments', []):
        comment_text = comment.get('comment', '')
        absa_result = predictABSA(absa_model, tokenizer, device, comment_text)
        sa_result = predictSA(sa_model, tokenizer, device, comment_text)
        comment['absa'] = absa_result
        comment['sa'] = sa_result[0]

# Save results to a new JSON file
with open(r'F:\code\datn\webjs\data\data-process.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

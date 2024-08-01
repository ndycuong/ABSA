from flask import Flask, request, jsonify
import torch
import sys  
import logging

from transformers import AutoTokenizer

app = Flask(__name__)

import torch
import torch.nn as nn
from transformers import AutoModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultitaskABSAConcat(nn.Module):
    def __init__(self, num_aspects=10, num_polarities=4):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2", output_hidden_states=True)
        self.dropout = nn.Dropout(0.2)
        self.aspect_classifiers = nn.ModuleList([
            nn.Linear(self.phobert.config.hidden_size * 4, num_polarities) for _ in range(num_aspects)
        ])
        self.num_aspects= num_aspects
        self.num_polarities= num_polarities

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask,  output_hidden_states=True)
        concat_hidden = torch.cat(outputs.hidden_states[-4:], dim=-1)  # Shape: (batch_size, seq_len, hidden_size*4)
        cls_output = concat_hidden[:, 0, :]  # Shape: (batch_size, hidden_size*4)
        cls_output = self.dropout(cls_output)
        
        # Calculate logits for each aspect
        aspect_logits = [classifier(cls_output) for classifier in self.aspect_classifiers]
        
        # Concatenate the aspect logits along the last dimension
        logits = torch.cat(aspect_logits, dim=1)
        logits = logits.view(-1, self.num_aspects, self.num_polarities)

        return logits
    
class MultibranchABSA(nn.Module):
    def __init__(self, num_aspects=11, num_polarities=4):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2", output_hidden_states=True)
        self.dropout = nn.Dropout(0.2)
        self.classifiers = nn.ModuleList([
            nn.Linear(self.phobert.config.hidden_size * 4, num_polarities) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        concat_hidden = torch.cat(outputs.hidden_states[-4:], dim=-1)  # Shape: (batch_size, seq_len, hidden_size*4)
        cls_output = concat_hidden[:, 0, :]  # Shape: (batch_size, hidden_size*4)
        cls_output = self.dropout(cls_output)
        logits = torch.stack([classifier(cls_output) for classifier in self.classifiers], dim=1)
        return logits


class MultitaskABSA(nn.Module):
    def __init__(self, n_aspects=11, n_labels=4):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2", output_hidden_states=True)  
        self.dropout = nn.Dropout(0.2)
        # Adjust the input features to linear layer since we are concatenating the last four layers
        self.linear1 = nn.Linear(self.phobert.config.hidden_size * 4, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, n_aspects * n_labels)
        self.n_aspects = n_aspects
        self.n_labels = n_labels

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # Tuple of hidden states
        # Concatenate the last four hidden layers
        concat_hidden = torch.cat(tuple(hidden_states[-i] for i in range(1, 5)), dim=-1)
        # We use the output of the last layer's [CLS] tokens
        concat_hidden = concat_hidden[:, 0, :]  # Shape: (batch_size, hidden_size * 4)
        x = self.linear1(self.dropout(concat_hidden))
        x = self.relu(x)
        logits = self.linear2(x)
        return logits.view(-1, self.n_aspects, self.n_labels)

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('vinai/phobert-base-v2', output_hidden_states=True)
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # Tuple of hidden states
        cls_output = hidden_states[:, 0]  # Taking the [CLS] token's representation
        output = self.drop(cls_output)
        logits = self.out(output) 

        return logits 

modelabsa_path1= r"E:\Downloads\webjs - Copy\python_server\model\Concat.pth"
absa_model1 = MultitaskABSAConcat(num_aspects=10, num_polarities=4)
absa_model1.load_state_dict(torch.load(modelabsa_path1, map_location=device))
absa_model1.to(device)

modelabsa_path2= r"E:\Downloads\webjs - Copy\python_server\model\MultibranchABSA.pth"
absa_model2 = MultibranchABSA(num_aspects=11, num_polarities=4)
absa_model2.load_state_dict(torch.load(modelabsa_path2, map_location=device))
absa_model2.to(device)

modelabsa_path3= r"E:\Downloads\webjs - Copy\python_server\model\Newapproach.pth"
absa_model3 = MultitaskABSA()
absa_model3.load_state_dict(torch.load(modelabsa_path3, map_location=device))
absa_model3.to(device)

modelsa_path = r"E:\Downloads\webjs - Copy\python_server\model\model4.pth"
sa_model=  SentimentClassifier(n_classes=3).to(device)
sa_model.load_state_dict(torch.load(modelsa_path, map_location=device))
# Load models and tokenizer here
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
# absa_model = torch.load("F:/slide/datn/model7.pth", map_location=device)
# sa_model = torch.load("F:/slide/datn/model4.pth", map_location=device)

def predictABSA(model, tokenizer, device, comments):
    if isinstance(comments, str):
        comments = [comments]  

    model.eval()
    inputs = tokenizer(comments, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)  # Outputs shape: (batch_size, num_aspects, num_polarities)
        predictions = torch.argmax(outputs, dim=2)  # Shape: (batch_size, num_aspects)
        predictions = predictions.cpu().numpy()  # Convert to numpy array for easier handling

    # Reverse mappings for aspects and sentiments
    sentiment_to_index = {'Positive': 1, 'Negative': 0, 'Neutral': 2, 'None': 3}
    index_to_sentiment = {v: k for k, v in sentiment_to_index.items()}
    aspect_to_index = {'GENERAL': 0, 'SER&ACC': 1, 'SCREEN': 2, 'CAMERA': 3, 'FEATURES': 4,
                       'BATTERY': 5, 'PERFORMANCE': 6, 'STORAGE': 7, 'DESIGN': 8, 'PRICE': 9, 'OTHERS': 10}
    index_to_aspect = {v: k for k, v in aspect_to_index.items()}

    # predicted_aspects = []
    result = {}

    # Iterate over each comment's predictions
    for comment_predictions in predictions:
        comment_aspects = []
        for aspect_idx, sentiment_idx in enumerate(comment_predictions):
            aspect = index_to_aspect[aspect_idx]
            sentiment = index_to_sentiment[sentiment_idx]
            if sentiment != 'None':  # Filter out 'None' sentiments
                result[aspect] = sentiment
    return result
    #         comment_aspects.append((aspect, sentiment))
    #     predicted_aspects.append(comment_aspects)

    # return predicted_aspects

def predictSA(model, tokenizer, device, comments):
    model.eval()
    inputs = tokenizer(comments, max_length=256, padding='max_length', truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        predictions = torch.argmax(probabilities, dim=1)  # Get predicted class
        predictions = predictions.cpu().numpy()  # Convert to numpy array for easier handling

    # Reverse mappings for sentiments
    index_to_sentiment = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    predicted_sentiments = [index_to_sentiment[idx] for idx in predictions]

    return predicted_sentiments

def calculate_score(sentiments):
    score = 0
    sentiment_score_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    for sentiment in sentiments:
        score += sentiment_score_map[sentiment]
    return score

def process_comments(comments):
    absa_results = []
    sa_results = []

    for i, comment in enumerate(comments):
        if 'comments' in comment:
            for sub_comment in comment['comments']:
                if 'comment' not in sub_comment:
                    rating_star = sub_comment.get('rating_star')
                    sentiment = 'Negative' if rating_star in [1, 2] else 'Neutral' if rating_star == 3 else 'Positive'
                    sa_results.append(sentiment)
                else:
                    absa_sentiments = predictABSA(absa_model1, tokenizer, device, sub_comment.get('comment'))
                    sa_sentiment = predictSA(sa_model, tokenizer, device, sub_comment.get('comment'))
                    absa_results.append(absa_sentiments)
                    sa_results.append(sa_sentiment)
        #     sentiment = 'Negative' if rating_star in [1, 2] else 'Neutral' if rating_star == 3 else 'Positive'
        #     sa_results.append(sentiment)
        # if not comment.get('comment'):
        #     rating_star = comment.get('rating_star')
        #     sentiment = 'Negative' if rating_star in [1, 2] else 'Neutral' if rating_star == 3 else 'Positive'
        #     sa_results.append(sentiment)
        # else:
        #     absa_sentiments = predictABSA(absa_model1, tokenizer, device, comment['comment'])[0]
        #     sa_sentiment = predictSA(sa_model, tokenizer, device, comment['comment'])[0]
        #     absa_results.append(absa_sentiments)
        #     sa_results.append(sa_sentiment)
    # Log the results for debugging
                logging.info(f"ABSA Results: {absa_results}")
                logging.info(f"SA Results: {sa_results}")

    return absa_results, sa_results

def calculate_sentiment_over_time(comments):
    df = pd.DataFrame(comments)
    df['ctime'] = pd.to_datetime(df['ctime'])
    df['sentiment'] = df['rating_star'].apply(lambda x: 'Negative' if x in [1, 2] else 'Neutral' if x == 3 else 'Positive')
    sentiment_over_time = df.groupby(df['ctime'].dt.to_period('M')).apply(lambda x: calculate_score(x['sentiment'].tolist())).reset_index(name='score')
    sentiment_over_time['ctime'] = sentiment_over_time['ctime'].astype(str)
    return sentiment_over_time.to_dict(orient='records')

logging.basicConfig(level=logging.INFO)
@app.route('/process_comments', methods=['POST'])
def process_comments_endpoint():
    try:
        data = request.json
        print(data)
        print(type(data))
        comments = data.get('comments', [])
        logging.info(f"Received {len(comments)} comments for processing.")
        
        # Log the received comments for debugging
        for i, comment in enumerate(comments):
            logging.info(f"Comment: {comment}")
            if 'comments' in comment:
                for sub_comment in comment['comments']:
                    if 'comment' not in sub_comment:
                        raise KeyError(f"'comment' key is missing in the comment data at index {i}")
                    logging.info(f"Processing sub_comment {i}: {sub_comment}")
            else:
                raise KeyError(f"'comments' key is missing in the main comment data at index {i}")

        absa_results, sa_results = process_comments(comments)

        absa_scores = [calculate_score(list(result.values())) for result in absa_results]
        sa_scores = [calculate_score([result]) for result in sa_results]

        response_data = {
            'absa_results': absa_results,
            'sa_results': sa_results,
            'absa_scores': absa_scores,
            'sa_scores': sa_scores,
        }
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        return jsonify({"error": "An error occurred"}), 500

@app.route('/predict/absa1', methods=['POST'])
def predict_absa1():
    try:
        data = request.json
        print(data)  # See what the data looks like
        print(type(data))  # Confirm it's a dictionary

        if 'text' not in data:
            return jsonify({"error": "Missing 'text' key in JSON request"}), 400
        text = data['text']
        predictions = predictABSA(absa_model1, tokenizer, device, text)
        return jsonify(predictions)
    except KeyError as e:
        print("JSON Key error: Missing 'text'", file=sys.stderr)
        return jsonify({"error": "Missing 'text' key in JSON request"}), 400
    except Exception as e:
        print("An error occurred: ", str(e), file=sys.stderr)
        return jsonify({"error": "An error occurred"}), 500
    
@app.route('/predict/absa2', methods=['POST'])
def predict_absa2():
    try:
        data = request.json
        print(data)  # See what the data looks like
        print(type(data))  # Confirm it's a dictionary

        if 'text' not in data:
            return jsonify({"error": "Missing 'text' key in JSON request"}), 400
        text = data['text']
        predictions = predictABSA(absa_model2, tokenizer, device, text)
        return jsonify(predictions)
    except KeyError as e:
        print("JSON Key error: Missing 'text'", file=sys.stderr)
        return jsonify({"error": "Missing 'text' key in JSON request"}), 400
    except Exception as e:
        print("An error occurred: ", str(e), file=sys.stderr)
        return jsonify({"error": "An error occurred"}), 500
    
@app.route('/predict/absa3', methods=['POST'])
def predict_absa3():
    try:
        data = request.json
        print(data)  # See what the data looks like
        print(type(data))  # Confirm it's a dictionary

        if 'text' not in data:
            return jsonify({"error": "Missing 'text' key in JSON request"}), 400
        text = data['text']
        predictions = predictABSA(absa_model3, tokenizer, device, text)
        return jsonify(predictions)
    except KeyError as e:
        print("JSON Key error: Missing 'text'", file=sys.stderr)
        return jsonify({"error": "Missing 'text' key in JSON request"}), 400
    except Exception as e:
        print("An error occurred: ", str(e), file=sys.stderr)
        return jsonify({"error": "An error occurred"}), 500

@app.route('/predict/sa', methods=['POST'])
def predict_sa():
    try:
        data = request.json
        print("Received data for SA:", data)  

        if 'text' not in data:
            return jsonify({"error": "Missing 'text' key in JSON request"}), 400
        text = data['text']
        predictions = predictSA(sa_model, tokenizer, device, text)
        print("Predictions:", predictions)  
        return jsonify({"sentiment": predictions})
    except KeyError as e:
        print("JSON Key error: Missing 'text'", file=sys.stderr)
        return jsonify({"error": "Missing 'text' key in JSON request"}), 400
    except Exception as e:
        print("An error occurred: ", str(e), file=sys.stderr)
        return jsonify({"error": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

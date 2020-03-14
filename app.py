import torch
from flask import Flask, jsonify, request, render_template
from transformers.modeling_bert import BertForMaskedLM, BertModel
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
import json


app = Flask(__name__)
tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese-whole-word-masking")
model = BertForMaskedLM.from_pretrained("bert-base-japanese-whole-word-masking")
model.eval()


def get_prediction(s):
    assert '[MASK]' in s
    input_ids = tokenizer.encode(s, return_tensors="pt")

    masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

    with torch.no_grad():
        result = model(input_ids)

    result_indices = torch.topk(result[0][:, masked_index], k=10).indices[0].tolist()
    return ' '.join([tokenizer.decode(i) for i in result_indices])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        s = request.form['sent']
        cloze = get_prediction(s)
        return cloze
    return render_template('/index.html')


if __name__ == '__main__':
    app.run()
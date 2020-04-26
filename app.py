import torch
from flask import Flask, request, render_template
from transformers.modeling_bert import BertForMaskedLM
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer


app = Flask(__name__)
tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese-whole-word-masking")
model = BertForMaskedLM.from_pretrained("bert-base-japanese-whole-word-masking")
model.eval()


def get_prediction(s):
    assert '[MASK]' in s
    input_ids = tokenizer.encode(s, return_tensors="pt")
    masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    with torch.no_grad():
        # (1, seq_len, vocab_size)
        logits, = model(input_ids)
        # (1, vocab_size)
        probs_for_mask = torch.softmax(logits[0, masked_index], dim=-1)
    topk_probs, topk_indices = torch.topk(probs_for_mask, k=10)
    return [tokenizer.decode([i]) for i in topk_indices.tolist()], topk_probs.tolist()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        s = request.form['sent']
        words, probs = get_prediction(s)
        return render_template('index.html', topk=zip(words, probs))
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

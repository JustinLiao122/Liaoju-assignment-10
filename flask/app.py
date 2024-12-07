from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from IPython.display import Image
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import open_clip


import pandas as pd
df = pd.read_pickle('image_embeddings.pickle')
df

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_highest(df, query_embedding, n=5):
    similarities = []

    for _, row in df.iterrows():
        dataset_embedding = row['embedding']  
        similarity = cosine_similarity(query_embedding, dataset_embedding)
        similarities.append((similarity, row['file_name']))

    top_results = sorted(similarities, key=lambda x: x[0], reverse=True)[:n]

    return top_results 


model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_path)



    image = preprocess(Image.open(temp_path)).unsqueeze(0)
    query_embedding = F.normalize(model.encode_image(image)).detach().numpy()[0]


    top_results = find_highest(df, query_embedding)
    os.remove(temp_path)

    return jsonify({
        'top_results': [
            {'file_name': file_name, 'similarity': float(similarity)}
            for similarity, file_name in top_results
        ]
    })




@app.route('/text', methods=['POST'])
def input_text():

    user_input = request.form.get("text", None)  
    if not user_input:
        return jsonify({'error': 'No text provided'}), 400

    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    text = tokenizer([user_input])
    query_embedding = F.normalize(model.encode_text(text)).detach().numpy()[0]


    top_results = find_highest(df, query_embedding)

    return jsonify({
        'text': user_input,
        'top_results': [
            {'file_name': file_name, 'similarity': float(similarity)}
            for similarity, file_name in top_results
        ]
    })
    


@app.route('/search', methods=['POST'])
def hybrid_query():
    text_input = request.form.get('text', None)
    file = request.files.get('file', None)
    lam = float(request.form.get('lam', 0.8))  

    if not text_input or not file:
        return jsonify({'error': 'Both text and image are required for hybrid query'}), 400

    temp_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_path)

    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text = tokenizer([text_input])
    text_query = F.normalize(model.encode_text(text))

    image = preprocess(Image.open(temp_path)).unsqueeze(0)
    image_query = F.normalize(model.encode_image(image))

    query_embedding = F.normalize(lam * text_query + (1.0 - lam) * image_query).detach().numpy()[0]

    os.remove(temp_path)

    top_results = find_highest(df, query_embedding)

    return jsonify({
        'top_results': [
            {'file_name': file_name, 'similarity': float(similarity)}
            for similarity, file_name in top_results
        ]
    })




@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('coco_images_resized/coco_images_resized', filename)

if __name__ == '__main__':
    app.run(debug=True)
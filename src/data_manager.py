import json
import numpy as np
import src.inference as inference

verbose = False

def get_words_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
                    
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("Invalid JSON format in the file.")

def update_embeddings(words_file_path, embedding_file_path):
    with open(embedding_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        embedding_count = len(lines)

    all_words = get_words_file(words_file_path)
    if embedding_count < len(all_words):
        np.set_printoptions(threshold=np.inf)
        with open(embedding_file_path, 'a', encoding='utf-8') as file:
            for i in range(embedding_count, len(all_words)):
                if verbose:
                    print('Inferencing',all_words[i]['lemma'],f'({i}/{len(all_words)})')
                word = all_words[i]['lemma']
                embedding = np.array2string(inference.get_embedding(word), max_line_width = np.inf)
                file.write(word + '#' + embedding + '\n')
        np.set_printoptions(threshold=6)

def get_all_embeddings(embedding_file_path):
    embeddings = []
    with open(embedding_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            word, embedding = line.split('#')
            embeddings.append((word, np.fromstring(embedding.strip('[]'), sep=' ')))
    return embeddings
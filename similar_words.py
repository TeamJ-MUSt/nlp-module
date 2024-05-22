import argparse

import src.inference as inference
import src.data_manager as data_manager

def parse_arguments():
    parser = argparse.ArgumentParser(description="Find top similar words.")
    parser.add_argument('words_path', type=str, help='Path to the file containing words')
    parser.add_argument('embedding_path', type=str, help='Path to the file containing word embeddings')
    parser.add_argument('query', type=str, help='The query word')
    parser.add_argument('count', type=int, help='Number of similar words to find')

    args = parser.parse_args()
    return args
def search_similar_words(query, count):
    result = []
    query_word = query
    query_embedding = next(filter(lambda x: x[0] == query_word, embeddings), None)[1]
    for word, embedding in embeddings:
        if word == query_word:
            continue
        similarity = inference.get_similarity(query_embedding, embedding)
        if len(result) < count or similarity > result[-1]['similarity']:
            result.append({'lemma' : word, 'similarity' : similarity})
            result.sort(key= lambda x:-x['similarity'])
            result = result[:count]
    return result

#ex query: "datas/word_db.txt" "datas/embeddings.txt" "教える" 3

if __name__ == "__main__":
    args = parse_arguments()
    words_file_path = args.words_path
    embedding_file_path = args.embedding_path

    data_manager.update_embeddings(words_file_path, embedding_file_path)
    embeddings = data_manager.get_all_embeddings(embedding_file_path)

    data_manager.verbose = True
    result = search_similar_words(args.query, args.count)
    print(result)

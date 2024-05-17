import src.inference as inference
import src.data_manager as data_manager

words_file_path = "word_db.txt"
embedding_file_path = "embeddings.txt"

data_manager.update_embeddings(words_file_path, embedding_file_path)

embeddings = data_manager.get_all_embeddings(embedding_file_path)

def search_similar_words(query, count):
    result = []
    query_word = query
    query_embedding = next(filter(lambda x: x[0] == query_word, embeddings), None)[1]
    for word, embedding in embeddings:
        if word == query_word:
            continue
        similarity = inference.get_similarity(query_embedding, embedding)
        if len(result) < count or similarity > result[-1]['similarity']:
            result.append({'word' : word, 'similarity' : similarity})
            result.sort(key= lambda x:x['similarity'])
    return result

result = search_similar_words('猫', 1)
print(result)
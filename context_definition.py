import src.multilingual_ai as ai
import argparse
import json
import re

# ex query : "空にある何かを見つめてたら@{#word#:#何#, #definitions#:[#일정하지 않은 것을 가리키는 대명사: 무엇.#, #아어 왜, 어째서, 무엇 때문에.#, #어째서#, #아니#, #어느 날#, #(내가 $아니$모르는) 무엇#, #어떤 것#]}@@空にある何かを見つめてたら@{#word#:#何#, #definitions#:[#일정하지 않은 것을 가리키는 대명사: 무엇.#, #(내가 $아니$모르는) 무엇#, #어떤 것#]}" 

def get_best_definition_index(sentence, word_data):
    word = word_data['word']
    definitions = word_data['definitions']
    
    def filter(text):
        text = re.sub(r'\(.*?\)', '', text)
        text = re.split(r'[:;,]', text)[-1]
        text = text.replace('.','').strip()
        return text
    definitions = [filter(text) for text in definitions]
    if word not in sentence:
        return 0
    replaced_sentences = [sentence.replace(word, definition) for definition in definitions]
    
    sentence_embedding = ai.get_embedding(sentence)
    replaced_embeddings = [ai.get_embedding(sent) for sent in replaced_sentences]

    similarities = [ai.get_similarity(sentence_embedding, embedding) for embedding in replaced_embeddings]
    best_definition_index = similarities.index(max(similarities))

    return best_definition_index



def parse_arguments():
    parser = argparse.ArgumentParser(description="Find the best matching meaning in context and returns index.")
    parser.add_argument('query', type=str, help='The sentence containing the word, A dictionary with word and its definitions. # is double quote, $ is single quote. sentence and word is splitted by @, each pair is splitted by @@')

    args = parser.parse_args()

    # Parse the JSON string into a dictionary
    try:
        queries = args.query.split('@@')
        pairs = []
        for query in queries:
            sentence, word = query.split('@')
            pairs.append([sentence.strip(), json.loads(word.strip().replace('#','\"').replace('$','\''))])

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        raise

    return pairs

if __name__ == "__main__":
    pairs = parse_arguments()
    indicies  = []
    for sentence, word in pairs:
        best_index = get_best_definition_index(sentence, word)
        indicies.append(best_index)

    print(indicies)

# nlp-module


## Environment
Python 3.9.12

## How to run
1. Clone this repository
```
git clone https://github.com/TeamJ-MUSt/nlp-module
cd nlp-module
```
2. Install `requirements.txt`
```
pip install -r requirements.txt
```
### context_definition.py
This returns the most accurate definition inside the context of a sentence.

usage: `python context_definition.py [-h] queries`

positional argument `queries`
- Each query is seperated by `@@`
- In each query, sentence and word dictionary is separated by `@`
- dictionary is a json formatted string, where `"` is replaced to `#` and `'` is replaced to `$`.

```
python context_definition.py "空にある何かを見つめてたら@{#word#:#何#, #definitions#:[#일정하지 않은 것을 가리키는 대명사: 무엇.#, #아어 왜, 어째서, 무엇 때문에.#, #어째서#, #아니#, #어느 날#, #(내가 $아니$모르는) 무엇#, #어떤 것#]}@@空にある何かを見つめてたら@{#word#:#何#, #definitions#:[#일정하지 않은 것을 가리키는 대명사: 무엇.#, #(내가 $아니$모르는) 무엇#, #어떤 것#]}" 
```

Outputs the list of definition indices for each query.

### similar_words.py
This will:

1. Update the embeddings file if it needs update (if there are less words than words file)
2. Search top X words and sorts by similarity
3. Outputs a list of dictionaries, where each dictionary has the word and similarity.

usage: python similar_words.py [-h] words_path embedding_path query count

positional arguments:  
- `words_path`: Path to the file containing words
- `embedding_path`: Path to the file containing word embeddings
- `query`: The query word
- `count`: Number of similar words to find

optional arguments:  
- `-h`, `--help`: Show help message  
```
python similar_words.py "datas/word_db.txt" "datas/embeddings.txt" "教える" 3
```

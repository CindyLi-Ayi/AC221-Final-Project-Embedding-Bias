# AC221-Final-Project-Embedding-Bias

## Metrics

### Bad word

```
from badword_matric import calculate_cos_with_badwords
res = calculate_cos_with_badwords(embedding={'word1':emb1, 'word2':emb2, ...})
## emb1 and 2 must be np.array
## res = {('male', 'female'):[x, y], ('he', 'she'):[x, y], ...}
## x is a list of cosine similarity of the m word and bad words that are in embedding
## y is a list of cosine similarity of the f word and bad words that are in embedding
```
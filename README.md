# AC221-Final-Project-Embedding-Bias

Yuanbiao Wang, Angel Hsu, Morris Reeves, Xinyi Li




## Motivation

Since text and images are two primary modes of communication in our society, it is of significant importance to explore bias in these two contexts, identify ways to mitigate bias in these two modes, and explore the potential connections between working with bias in the two modes. Commonly, learned representations, the numerical vector representations of either text or images learned from large pre-trained neural networks, are used to perform transfer learning on downstream tasks. As such models increase in complexity, it is increasingly important to examine the biases in these unexplainable vector representations, and that will be our focus in this project.

In this project, we prioritized exploring bias in text by (1) training our own embeddings to identify where bias may originate from, (2) identifying whether existing embeddings have bias, and (3) using biased embeddings for classification to identify whether bias in embeddings diffuse to downstream tasks. We also supplemented this analysis by exploring cross-modal representations produced by CLIP, a self-supervised learning algorithm that aims to acquire cross-modal representations by contrasting co-occurring texts and images. CLIP was introduced by OpenAI in 2021; the algorithm is garnering attention in many fields, including text-prompted image generation, visual question answering, and image captioning. While this new algorithm shows great potential in bridging semantic understanding between images and text, it also embodies new ethical risks by introducing risks from both areas.




## Definition and Metrics

For this project, we primarily focus on gender bias and we define bias using the following 2 metrics.

### Cosine similarity with offensive words
#### Definition
We have a list of male word and female word listed below.
```python
male_words = ['he', 'male', 'man', 'father', 'boy', 'husband']
female_words = ['she', 'female', 'woman', 'mother', 'girl', 'wife']
```
We also have a list of offensive/profane word from CMU: https://www.cs.cmu.edu/~biglou/resources/

So we define the bias using the cosine similarity between 2 embedding vectors. If a gender-related word has higher cosine similarity with an offensive word compared to the corresponding word for the opposite gender, then we believe the embedding has a bias against that gender. We take the mean cosine similarity defined as follows to aggregate the performance over the offensive word list.

![badwords_definition](/images/badwords_definition.png)

#### Usage 

```python
from badword_matric import calculate_cos_with_badwords
res = calculate_cos_with_badwords(embedding={'word1':emb1, 'word2':emb2, ...})
## emb1 and 2 must be np.array
## res = {('male', 'female'):[x, y], ('he', 'she'):[x, y], ...}
## x is a list of cosine similarity of the m word and bad words that are in embedding
## y is a list of cosine similarity of the f word and bad words that are in embedding
```

### WEAT score
#### Definitiion
As implied by its name, the WEAT (Word Embedding Association Test) score measured bias in associations. Introduced by Caliskan et al. (2016), the WEAT score attempts to capture the strength of the association between two sets of target words (e.g. {math, science} and {art, literature}) and two sets of attribute words (e.g. {male, man} and {female, woman}). Intuitively, the score asks: is male/man more associated with math/science than art/literature, relative to female/woman? It does so through comparisons of mean cosine similarities:

![weat_def1](/images/weat_definition_1.png)
![weat_def2](/images/weat_definition_2.png)

We use the same word lists in Caliskan et al. for the comparisons (Math vs. Arts) and (Science vs. Arts), to avoid the risk of manipulating our results to achieve certain conclusions. To compute the WEAT score, we use the **responsibly** package ([link](https://github.com/ResponsiblyAI/responsibly)), following the demos at the [docs](https://docs.responsibly.ai/notebooks/demo-word-embedding-bias.html).

## Tasks

### Bias in Embeddings trained from scratch

We explore the sources of bias (as measured by the 2 metrics above) by training our own embeddings using various configurations of:

* Data source (Twitter, Reddit, CNN/DailyMail)
* Dataset size (10k, 15k, 20k, 25k, 30k documents)

For word2vec model training, we use the CBOW implementation from the **gensim** package, with parameters: context window size of 5, minimum word count of 5, learning rate of 0.01, and 15 epochs. After training word2vec using each of the combinations of configurations (of data source and dataset size above), we compute the two bias metrics above. 

Please see the *word2vec_training.ipynb* notebook for the associated code and additional details.

### Bias in Existing Word Embeddings

We think it is very important to examine the bias in existing embeddings since people too often just use these pretrain embeddings and trust them in their performance as well as fairness. At the same time, we hope to find clues of where the bias could come from by doing comparative studies across different embeddings and also their variants, and also give guidance on which is a less biased embedding when fairness is an important concern of the project. 

In this task, we examined 3 types of common embeddings - Glove, Word2vec and ELMo. As a summary of our findings, we think there are definitely bias in these pretrained word embeddings, but their level of bias varies. Therefore, the users have the choice to use a less biased one in their training. Most importantly, they should try to avoid the embeddings trained on informal dataset like Twitter. Other less significant factors include embedding dimention, dataset size and embedding layers depth. Also, if people want to train embeddings themselves, they may also want to be carefully when choosing which corpus to train on as well as the value of these other factors.

You can find more details about this task in `task2-bias_in_pretrained_word_embeddings.ipynb`. You will need to first download the embeddings with

```python
./get_data.sh
```



### Whether bias in embeddings diffuses into downstream task (Sentiment Analysis)

### Bias in Image Embedding (CLIP)


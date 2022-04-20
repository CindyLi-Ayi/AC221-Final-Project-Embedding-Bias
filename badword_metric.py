import numpy as np

male_words = ['he', 'male', 'man', 'father', 'boy', 'husband']
female_words = ['she', 'female', 'woman', 'mother', 'girl', 'wife']

bad_words_path = "./data/bad-words.txt"
with open(bad_words_path, 'r') as file:
    bad_words = [i[:-1] for i in file.readlines()][1:]
    
def filter_vocab(vocab, words):
    filtered = [word for word in words if word in vocab.keys()]
    print(f"original words: {len(words)}, in vocab words: {len(filtered)}")
    return filtered

def cos_similarity(w1, w2, embedding):
    try:
        e1 = embedding[w1]
        e2 = embedding[w2]
        return (e1*e2).sum()/np.sqrt((e1*e1).sum()*(e2*e2).sum())
    except:
        for w in [w1, w2]:
            if w not in embedding.keys():
                print(f"{w} not in vocab")
        return np.nan

def calculate_cos_with_badwords(embedding):
    res = {}
    bad_words_invocab = filter_vocab(embedding, bad_words)
    for m, f in zip(male_words, female_words):
        x = [cos_similarity(w, m, embedding) for w in bad_words_invocab]
        y = [cos_similarity(w, f, embedding) for w in bad_words_invocab]
        res[(m,f)] = (x, y)
    return res


if __name__=="__main__":
    print(calculate_cos_with_badwords({'male':np.array([0,1]), 'female':np.array([0,1]), 'crime':np.array([1,0])}))
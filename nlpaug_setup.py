import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nlpaug.util.file.download import DownloadUtil
dest_dir ='./nlp_model'
DownloadUtil.download_word2vec(dest_dir=dest_dir) # Download word2vec model
DownloadUtil.download_glove(model_name='glove.6B', dest_dir=dest_dir) # Download GloVe model
DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir=dest_dir) # Download fasttext model
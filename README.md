# Effect of Combination of HBM and Certainty Sampling on Workload of Semi-Automated Grey Literature Screening 

This repository is temporarily associated with paper [Lu, J., Henchion, M. and Mac Namee, B., 2021. Effect of Combination of HBM and Certainty Sampling on Workload of Semi-Automated Grey Literature Screening](https://icml.cc/Conferences/2021/ScheduleMultitrack?event=8362) (to be published at the 3rd Workshop of Human in the Loop, 38<sup>th</sup> ICML, 2021)

## Grey Literature Datasets (to be released)
Csv files in the directory is just for providing a toy experiment samples. The results of this dataset is invalid since the dataset here is used for pretraining the language model (therefore performance will be super high even if only 10 example are used for training).

## Usage

### Dependencies
Tested Python 3.6, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.19.5](http://www.numpy.org/)
* Required: [scikit-learn >= 0.21.1](http://scikit-learn.org/stable/)
* Required: [pandas >= 1.1.5](https://pandas.pydata.org/)
* Required: [gensim >= 3.7.3](https://radimrehurek.com/gensim/)
* Required: [matplotlib >= 3.3.3](https://matplotlib.org/)
* Required: [torch >= 1.9.0](https://pytorch.org/)
* Required: [transformers >= 4.8.2](https://huggingface.co/transformers/)
* Required: [Keras >= 2.0.8](https://keras.io/)
* Required: [Tensorflow >= 1.14.0](https://www.tensorflow.org/)
* Required: [FastText model trained with Wikipedia 300-dimension](https://fasttext.cc/docs/en/pretrained-vectors.html)
* Required: [GloVe model trained with Gigaword and Wikipedia 200-dimension](https://nlp.stanford.edu/projects/glove/)
* Required: packaging >= 20.0


### Step 1. Data Processing

The first step is encoding raw text data into different high-dimensional vectorised representations. The raw text data should be stored in directory "corpus_data/", each dataset should have its individual directory, for example, the "animal_by_product/" directory under folder "corpus_data/". The input corpus of documents should consist of plain text files stored in csv format (two files for one corpus, one for documents belong to class A and one for documents for class B) with a columan named as __text__. It should be noted that the csv file must be named in the format _#datasetname_neg_text.csv_ or _#datasetname_pos_text.csv_. Each row corresponding to one document in that corpus, the format can be refered to the csv file in the sample directory "corpus_data/animal_by_product/". Then we can start preprocessing text data and converting them into vectors by:

    python encode_text.py -d dataset_name -t encoding_methods
    
The options of -t are `hbm` (corresponding to the sentence representation generated by the token-level RoBERTa encoder in the paper), `roberta-base`, `fasttext`, and `PV-TD` for example `-t roberta-base,fasttext` means encoding documents by RoBERTa and FastText respectively. The encoded documents are stored in directory "dataset/", e.g. the FastText document representations are stored in "dataset/fasttext/" and HBM and RoBERTa embeddings are stored in "dataset/roberta-base/". It should be noted that the sentence representations for hbm is suffixed by ".pt" and the document representations generated by RoBERTa are suffixed by ".csv"(average all tokens to represent a document) or "\_cls.csv" (using classifier token "\<s\>" to represent a document) For encoding by FastText and PV-TD, you need to download the pretrained FastText model in advance (see __Dependencies__).

### Step 2. Run HBM + Certainty/Uncertainty/FastRead

We can evaluate the HBM + Certainty/Uncertainty:

    python run_active_learning_hbm.py -l num_active_learning_iterations -r random_seeds -s selection_strategy -m max_len

The `num_active_learning_iterations` can be random int numbers up to the size of the datasets, for example  `-l 10` means active learning process iterate 10 loops and 100 documents will be shown to the user (we set the number of labelled examples each step 10 documents).  The `random_seeds` are random state for subsampling initial training set. For example `-r 1988,1999` will training HBM with 2 different training sets, i.e. 10 labelled instances sampled by seed 1988, 10 labelled instances sampled by seed 1999. `max_len` is the max length of sentences for document set for HBM and `selection_strategy` means the selection method used in active learning, the options are `mostConfident` for certainty sampling and `uncertainty` for uncertainty sampling. Additionally, if you want to output attention scores as well, you need `-a` otherwise `-n` for disabling output attention scores.

For HBM + FastRead

    python run_active_learning_hbm_fastread.py

### Step 2. Run SVM-based method (RoBERTa, PV-TD) + Certainty/Uncertainty/FastRead

The script then evaluate the performance of HBM in the rest testing set (i.e. the whole dataset minus the 200 instances that sampled out as the training set, the details can be referred in the paper). The evaluation results are stored in directory "outputs/". Furthermore, the concrete results of each step are stored in "outputs/hbm_results/". The results files starting with "auc_" store the AUC score results while files starting with "raw_" store the confusion matrix (tp, tn, fp, fn).


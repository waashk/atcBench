# atcBench: A thorough benchmark of automatic text classification 

## From traditional approaches to large language models

This repository contains a Python 3 implementation for atcBench.

## Setup environment

Clone this repository in your machine. Execute the installation under the settings directory.

```
git clone https://github.com/waashk/atcBench.git
```

You have two options for installing: virtualenv or Docker.

### Installing and Activating through virtualenv

```
cd atcBench/settings/
bash setupENV.sh
```

For activating the environment:

```
source env/bin/activate
```

### Installing and Activating through Dockerfile


```
cd atcbench/settings/
docker build -t atcbench:1.0 .
```

For activating the Docker:

```
cd atcbench/
docker run --gpus all --ipc=host --ulimit memlock=-1 --rm --name atcbench -p 8888:8888 -v `pwd`:/home/atcbench -i -t atcbench:1.0 /bin/bash
```

## Requirements

This project is based on ```python==3.9```, and it requires ```Docker version >= 24.0.1``` (if installed through Docker).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Examples

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TB_CONFIG_NAME="config_default.yaml"
python main.py 
```

## Automatic Text Classification Datasets

All datasets in .parquet format are provided in the following link: [LINK](https://huggingface.co/datasets/waashk/)

| **Dataset**  | **Size** | **Dim.** | **# Classes** | **Density** | **Skewness**                                                |
|--------------|----------|----------|---------------|-------------|----------------------|------------------------------------------------|
| DBLP         | 38,128   | 28,131   | 10            | 141         | Imbalanced            |
| Books        | 33,594   | 46,382   | 8             | 269         | Imbalanced            |
| ACM          | 24,897   | 48,867   | 11            | 65          | Imbalanced            |
| 20NG         | 18,846   | 97,401   | 20            | 96          | Balanced              |
| OHSUMED      | 18,302   | 31,951   | 23            | 154         | Imbalanced            |
| Reuters90    | 13,327   | 27,302   | 90            | 171         | Extremely Imbalanced  |
| WOS-11967    | 11,967   | 25,567   | 33            | 195         | Balanced              |
| WebKB        | 8,199    | 23,047   | 7             | 209         | Imbalanced            |
| Twitter      | 6,997    | 8,135    | 6             | 28          | Imbalanced            |
| TREC         | 5,952    | 3,032    | 6             | 10          | Imbalanced            |
| WOS-5736     | 5,736    | 18,031   | 11            | 201         | Balanced              |
| SST1         | 11,855   | 9,015    | 5             | 19          | Balanced              |
| pang_movie   | 10,662   | 17,290   | 2             | 21          | Balanced              |
| Movie Review | 10,662   | 9,070    | 2             | 21          | Balanced              |
| vader_movie  | 10,568   | 16,827   | 2             | 19          | Balanced              |
| MPQA         | 10,606   | 2,643    | 2             | 3           | Imbalanced            |
| Subj         | 10,000   | 10,151   | 2             | 24          | Balanced              |
| SST2         | 9,613    | 7,866    | 2             | 19          | Balanced              |
| yelp_reviews | 5,000    | 23,631   | 2             | 132         | Balanced              |
| vader_nyt    | 4,946    | 12,004   | 2             | 18          | Balanced              |
| agnews       | 127,600  | 39,837   | 4             | 37          | Balanced              |
| yelp_2013    | 335,018  | 62,964   | 6             | 152         | Imbalanced            |
| imdb_reviews | 348,415  | 115,831  | 10            | 326         | Imbalanced            |
| sogou        | 510,000  | 98,974   | 5             | 588         | Balanced              |
| medline      | 860,424  | 125981   | 7             | 77          | Imbalanced            |

To guarantee the reproducibility of the obtained results, all datasets and their respective CV train-test partitions are available on in the respective LINK column.

Each dataset contains the following files:
- data.parquet: pandas DataFrame with texts and associated labels for each document.
- splits/split_\<k\>.pkl:  pandas DataFrame with k-cross validation partition.
- tfidf/: the TFIDF representation for each fold in the CSR matrix format. (.gz)

## Output

The outputs of the main.py script are:

- saida_\<method\>.json: A JSON file containing general information about the execution, including i. reduction, ii. time, iii. hyperparameters, among others.
- measures.fold_<k>.json: Includes micro-f1, macro-f1, the confusion matrix, time_train, time_predict measures, and the machine name you used to generate the results for this specific fold k.
- pred.fold_<k>.parquet: A pandas dataframe containing y_pred and y_test for this specific fold k.
- proba.fold_<k>.gz: A numpy matrix generated by dump_svmlight_file(zero_based=False) and compressed with gzip, containing the posterior probabilities provided by the classifier applied to specific fold k.

## Citation

```
@article{cunha2025arxiv,
  title={A thorough benchmark of automatic text classification: From traditional approaches to large language models},
  author={Cunha, Washington and Rocha, Leonardo and Gon{\c{c}}alves, Marcos Andr{\'e}},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```


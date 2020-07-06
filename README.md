# RETRO: Relation Retrofitting For In-Database Machine Learning on Textual + Numerical Data
RETRO is a framework that provides tools to automatically extract text values from a PostgreSQL database, represent those text values by a continuous vector representation using a word embedding model.
In order to incorporate semantic knowledge from the database into the representation, it extracts additional relational information from the database and uses this knowledge to refine the embeddings by a relational retrofitting method.
The resulting embeddings can then be used to perform machine learning tasks.

For more details about the algorithm, take a look at the [paper](https://arxiv.org/abs/1911.12674).

## Setup

In order to connect to a PostgreSQL database, you have to configure the database connection in the `config/db_config.json`.
In order to run RETRO you might need to install some python packages (e.g. numpy, psycopg2, networkx, scipy, sklearn, word2number).

### Import Word Embedding Model
RETRO is independent of the word embedding model you want to use for the text values.
If you want to use the popular Google News word embedding model, you can download it and import it to the database as follows (requires the [gensim](https://radimrehurek.com/gensim/) python package):
```
mkdir vectors
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" -P vectors
gzip --decompress vectors/GoogleNews-vectors-negative300.bin.gz
python3 core/transform_vecs.py
python3 core/vec2database.py config/google_vecs.config config/db_config.json
```

### Execute RETRO

In order to execute RETRO, you might reconfigure `config/retro_config.json`.
For example, in the `TABLE_BLACKLIST` property, you can define tables that should be ignored by the retrofitting.
You can start the retrofitting by executing the following commands:
```
mkdir output
python3 core/run_retrofitting.py config/retro_config.json config/db_config.json
```
After the script is passed through, you can find the results in the table `retro_vecs` in the PostgreSQL database and the file `output/retrofitted_vectors.wv`.

## Example ML Task

As one example of a machine learning task, you can run a category imputation task on a dataset of [Open Food Facts](https://www.kaggle.com/openfoodfacts/world-food-facts).

### Create the Database

The script provided in [open-food-facts-dataset-import](https://github.com/guenthermi/open-food-facts-postgresql-import) can be used to create a database containing the data from the Open Food Facts dataset.
In order to set up the database, clone the repository and follow the instruction in the README file.

### Configuration

The configuration files `config/db_config.json` and `ml/db_config.json` have to be adapted to your database.
By using the default configuration in `config/retro_config.json` the category and genre property are ignored since those are the properties we want to predict with our classifier (often the genre property equals the category).

Numeric Encodings in `config/retro_config.json`:

| MODE                    | description                                                                                                               | used settings                                           |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| disabled                | numeric values aren't encoded at all                                                                                      | -                                                       |
| random                  | vector contains random numbers between -1.0 and 1.0                                                                       | -                                                       |
| one-hot                 | one-hot encoding with -1 and 1                                                                                            | NORMALIZATION, BUCKETS, NUMBER_DIMS                     |
| one-hot-gaussian        | one-hot encoded vector with  gaussian filter applied                                                                      | NORMALIZATION, BUCKETS, NUMBER_DIMS, STANDARD_DEVIATION |
| one-hot-column-centroid | centroid between one-hot encoded vector and we-vector of the column name                                                  | NORMALIZATION, BUCKETS                                  |
| we-regression           | uses word embeddings of neighboring numeric values to interpolate vector                                                  | NORMALIZATION                                           |
| unary                   | unary encoding with -1 and 1                                                                                              | NORMALIZATION, BUCKETS, NUMBER_DIMS                     |
| unary-gaussian          | unary encoded vector with gaussian  filter applied                                                                        | NORMALIZATION, BUCKETS, NUMBER_DIMS, STANDARD_DEVIATION |
| unary-column-centroid   | centroid between unary encoded vector and we-vector of the column name                                                    | NORMALIZATION, BUCKETS                                  |
| unary-column-partial    | the first NUMBER_DIMS dimensions are used for unary encoding values, the rest for we-vector values of the column name     | NORMALIZATION, BUCKETS, NUMBER_DIMS                     |
| unary-random-dim        | the number encoding vector indexes are randomly distributed with the column name used as seed                             | NORMALIZATION, BUCKETS, NUMBER_DIMS                     |

Additional Settings:
* NORMALIZATION (Boolean, default: True): Vectors get normalized to length 1
* NUMBER_DIMS (Integer from 0 to 300, default: 300): Number of dimensions used for numeric encoding
* BUCKETS (Boolean, default: True): If true, all values get evenly distributed over all NUMBER_DIMS buckets, otherwise the values are divided in 300 equally sized numeric ranges between the min and max of the column
* STANDARD_DEVIATION (Float, default: 1.0): Represents the corresponding parameter of the gaussian function

### Retrofitting

First, relational retrofitting has to be executed on the database.
Given a word embedding model is imported to the database as described above, retrofitting can be executed as follows:

```
mkdir output
python3 core/run_retrofitting.py ml/retro_config.json ml/db_config.json
```

### Run Classification

After executing the retrofitting, the classification can be started.
The category imputation task can be performed with `ml/multi_class_prediction.py`:

```
python3 ml/multi_class_prediction.py ml/db_config.json ml/classify_food_categories_config.json
```

After running and evaluating the classification, this should output a results file and a box-plot diagram in the output folder defined in `ml/classify_food_categories_config.json` (default: `output/`).

# Kronos

## Code Examples

# Produce NER training data from the timetable

# Use NER Annotator to annotate timetable cell data https://tecoholic.github.io/ner-annotator/

# Split full NER annotation data into train and dev sets
```sh
poetry run python -m kronos.pipelines.split_ner_annotations -pna ./data/02_intermediate/timetable_annotations.json -pta ./data/02_intermediate/train_timetable_annotations.json -pda ./data/02_intermediate/dev_timetable_annotations.json
```

# Parse JSON formatted annotations into DocBin formatted annotations

```sh
poetry run python -m kronos.pipelines.convert_jsons_to_docbins -ptj ./data/02_intermediate/train_timetable_annotations.json -pdj ./data/02_intermediate/dev_timetable_annotations.json -ptd ./data/03_primary/train_timetable_annotations.spacy -pdd ./data/03_primary/dev_timetable_annotations.spacy
```

# Populate basic NER training configuration file with default values

```sh
poetry run python -m spacy init fill-config ./data/03_primary/base_config.cf
g ./data/03_primary/config.cfg
```

# Train a customer NER component in a spacy pipeline

```sh
poetry run python -m spacy train ./data/03_primary/config.cfg --output ./data/03_primary/ --paths.train ./data/03_primary/train_timetable_annotations.spacy --paths.dev ./data/03_primary/dev_timetable_annotations.spacy
```

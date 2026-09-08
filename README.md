# Sentiment Analysis of AdaKami Online Lending Application Reviews: Data and Code

Replication package for the study comparing Naive Bayes Classifier (NBC), Support Vector Machine (SVM), and Random Forest (RF) for sentiment classification of Indonesian peer-to-peer lending application reviews.

**Authors:** Avini Fazrie, Sfenrianto
**Affiliation:** Information Systems Management Department, BINUS Graduate Program, Bina Nusantara University, Jakarta, Indonesia
**Related publication:** *Sentiment Analysis of AdaKami Online Lending Application Reviews Using Comparative Machine Learning*, ICIMTech 2026
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0), full text in [`LICENSE`](LICENSE)

---

## Contents

```
data/
  ulasan_adakami_2025_raw.csv           35,499 rows  raw scraped reviews
  ulasan_adakami_2025_preprocessed.csv  32,905 rows  after 8-stage preprocessing
  ulasan_dengan_prediksi.csv            32,905 rows  with NBC model predictions
  anotasi_200_ulasan.csv                   200 rows  manual annotation, 2 annotators
  anotasi_200_dengan_rating.csv            200 rows  annotation with star ratings
  frekuensi_kata_negatif.csv                20 rows  top negative-review terms

notebooks/
  step1_scraping.ipynb          data collection via google-play-scraper
  step2_eda.ipynb               exploratory data analysis
  step3_labeling_kappa.ipynb    label validation, Cohen's kappa
  step4_preprocessing.ipynb     8-stage text preprocessing
  step5_modeling.ipynb          TF-IDF, SMOTE, GridSearchCV, 3 classifiers
  step6_evaluation.ipynb        full-corpus prediction, temporal and topic analysis

models/
  model_nbc.pkl                 MultinomialNB, alpha = 0.5 (best model)
  model_svm.pkl                 LinearSVC, C = 0.1
  model_rf.pkl                  RandomForest, n_estimators = 200, max_depth = 20
  tfidf_vectorizer.pkl          fitted TF-IDF vectorizer, 10,000 features

dashboard.py                    Streamlit visualization dashboard
requirements.txt                pinned dependencies
LICENSE                         CC BY 4.0 legal code
```

`output/figures_en/` is not included in this deposit. It is created automatically by
Steps 5 and 6 when you run them, and holds the generated figures.

## Data source and scope

User reviews of the AdaKami application (application ID `com.adakami.dana.kredit.pinjaman`) published on the Google Play Store between January and December 2025, collected in Indonesian only, with no rating filter applied.

## Privacy notice

The reviews in this dataset are public user-generated content, posted voluntarily on the Google Play Store and readable by anyone.

The scraper originally captured a `username` field. That column has been removed from every file here. It was never used in any analysis step, model, or figure. What remains is review text, star rating, date, non-identifying metadata, and sentiment labels.

All 35,499 reviews were screened for phone numbers, email addresses, national identity numbers, and bank account numbers. None were found.

A small number of reviews contain a name the reviewer typed themselves, usually while stating that a loan was not taken out in their name. This text is part of the public review as published and has been left unchanged, so that the analysis remains reproducible. If you are the author of such a review and would like it removed, contact the corresponding author and it will be taken out of future versions of this deposit.

The `review_id` field is an opaque identifier assigned by Google Play. It is kept for deduplication and traceability and does not by itself identify a person.

This handling follows Indonesian Law No. 27 of 2022 on Personal Data Protection, Article 15(1)(e), which provides an exemption for statistical and scientific research purposes.

## Reproducing the results

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order, `step1` through `step6`. They resolve `../data/` and `../models/` relative to their own location, so open them from inside `notebooks/` and leave the directory layout above unchanged.

Step 1 re-scrapes from the live Play Store and will therefore return a different corpus; to reproduce the published figures exactly, skip Step 1 and start from `data/ulasan_adakami_2025_raw.csv`. Step 1 is also the one notebook written for Google Colab: it writes to the working directory and downloads the result via `google.colab.files`, so if you run it locally, move its output into `data/` yourself.

Run the dashboard from the repository root, where it reads `data/ulasan_dengan_prediksi.csv`:

```bash
streamlit run dashboard.py
```

## Key results

| Model | CV F1-macro | Accuracy | Precision | Recall | F1-macro |
|-------|-------------|----------|-----------|--------|----------|
| NBC (alpha = 0.5) | 0.6318 | 0.8221 | 0.6386 | 0.7021 | **0.6416** |
| SVM (C = 0.1) | 0.6237 | 0.8164 | 0.6344 | 0.6786 | 0.6299 |
| RF (n = 200, depth = 20) | 0.6214 | 0.8240 | 0.6166 | 0.6272 | 0.6134 |

Majority-class baseline: accuracy 0.6485, F1-macro 0.2623.

Inter-annotator agreement on the 200-review validation sample: Cohen's kappa = 0.8595 (almost perfect).

Wilcoxon signed-rank test on per-fold F1-macro, NBC versus SVM: W = 0, p = 0.0625. The difference is not statistically significant at alpha = 0.05. With only five paired folds, 0.0625 is the smallest attainable p-value when all differences share the same sign, so this test sits at the limit of its resolution.

Predicted sentiment distribution across all 32,905 reviews: positive 19,061 (57.9 percent), negative 10,030 (30.5 percent), neutral 3,814 (11.6 percent).

## Notes on the model files

The `.pkl` files are Python pickles produced with the scikit-learn version pinned in `requirements.txt`. Loading pickles executes code, so load them only from a source you trust. All models can be regenerated from `step5_modeling.ipynb` without using these files.

## Dashboard

An interactive version of the results is deployed at
https://adakami-sentiment-dashboard-bz7nyyzveyq4u7tztklz9s.streamlit.app/
with source code at
https://github.com/avinavini/adakami-sentiment-dashboard

## License

The data and code in this repository are released under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0). The full legal code is in
[`LICENSE`](LICENSE); a plain-language summary is at
https://creativecommons.org/licenses/by/4.0/

Copyright (c) 2026 Avini Fazrie and Sfenrianto.

You are free to share and adapt this material, including commercially, provided
you give appropriate credit, link to the licence, and indicate whether changes
were made. Attribute as:

> Fazrie, A. and Sfenrianto (2026). *Sentiment Analysis of AdaKami Online Lending
> Application Reviews: Data and Code.* Licensed under CC BY 4.0.

The licence covers this deposit. It does not extend to the underlying review
text, which remains the work of the Google Play users who wrote it, nor to any
third-party dependency listed in `requirements.txt`, which carries its own
licence.

## Citation

If you use this dataset or code, please cite the accompanying ICIMTech 2026 paper and this deposit.

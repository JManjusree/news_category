# News Category Classification

## Project Description

This project is used to **classify news articles into different categories** using Machine Learning.

The dataset contains news articles with categories such as **Politics, Business, Entertainment, Sports, Travel, Food & Drink, Technology**, etc.

## Dataset

* Dataset: News Category Dataset
* Input: News headline and short description
* Target: News category

## Preprocessing

The following preprocessing steps were performed:

* Checked the dataset
* Handled missing values
* Removed duplicate records
* Combined `headline` and `short_description`
* Split the data into training and testing data
* Used **TF-IDF** for converting text into numerical features

## Models Tested

Different Machine Learning models were tested and compared.

The main models/experiments included:

* Logistic Regression
* Naive Bayes
* Linear SVM
* Balanced SVM
* SVM hyperparameter tuning

After comparison, **Linear SVM** was selected as the final model.

## Final Model

The final model is:

```text
Linear SVM
C = 0.5
```

Final accuracy:

```text
62.42%
```

## Project Structure

```text
news-category-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── final_svm_model.pkl
│
├── notebooks/
│   └── news.ipynb
│
├── outputs/
│   └── confusion_matrix.png
│
├── reports/
│   └── classification_report.txt
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
└── requirements.txt
```

## How to Run

First run preprocessing:

```bash
python src/preprocess.py
```

Then train the model:

```bash
python src/train.py
```

Finally, evaluate the model:

```bash
python src/evaluate.py
```

## Results

The final Linear SVM achieved approximately **62.42% accuracy** on the test dataset.

The classification report and confusion matrix are also included in the project.

## Conclusion

In this project, different Machine Learning approaches were tested for news category classification. TF-IDF was used to convert the news text into numerical features. After comparing the models, **Linear SVM with C = 0.5** gave the best overall performance and was selected as the final model.

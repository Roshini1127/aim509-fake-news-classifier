# Dataset Instructions

The training program expects a file named news.csv in this folder.

The CSV must contain two columns:

- text: the news headline or article text
- label: the known label for that article, such as FAKE or REAL

Example format only:

text,label
"Example article text for testing the file format",FAKE
"Another example article for testing the file format",REAL

These example rows are illustrative only. They are not a real dataset and must not be used to report model accuracy.

## Dataset requirements

1. Use a properly labelled dataset suitable for academic use.
2. Check the dataset's licence and record its original source.
3. Remove empty rows and inspect duplicate articles.
4. Document how the labels were assigned.
5. Evaluate the model on a held-out test set.

The model predicts dataset labels from text patterns. Its predictions do not independently prove that a news story is true or false.

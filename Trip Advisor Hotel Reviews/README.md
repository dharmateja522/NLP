# Trip Advisor Hotel Reviews Analysis

## Problem Statement

### Background Information

Trip Advisor is a popular travel website where users can write reviews about hotels, restaurants, and other travel-related services. These reviews offer valuable insights into the quality, customer experience, and overall satisfaction of various establishments. Analyzing these reviews can help businesses understand customer preferences, improve their services, and make informed decisions.

### Dataset Information

The dataset used in this project consists of Trip Advisor hotel reviews, stored in a CSV file format. It contains the following columns:
- **Review**: The text of the reviews written by customers.
- **Rating**: The rating given by customers (on a scale of 1 to 5).
- **Length**: The length of each review.

## Problem Statement

The objective of this project is to perform exploratory data analysis (EDA) on the Trip Advisor hotel reviews dataset. The analysis aims to gain insights into the sentiment, length, word count, and other characteristics of the reviews. By analyzing the reviews, we aim to understand customer sentiments, identify patterns or trends, and extract useful information to improve the quality of services provided by hotels.

## Approach

1. **Importing Required Libraries**
   - Import libraries such as NumPy, Pandas, Matplotlib, Seaborn, NLTK, and other relevant modules for data manipulation, visualization, and natural language processing tasks.

2. **Loading the Dataset**
   - Read the Trip Advisor hotel reviews dataset from a CSV file using `pd.read_csv()`.
   - Print the first few rows of the DataFrame to inspect its structure and content.

3. **Transforming Rating into Binary Labels**
   - Transform the 'Rating' column into binary labels ('Positive' and 'Negative') based on a predefined threshold. Ratings greater than 3 are mapped to 'Positive', while ratings between 1 and 3 are mapped to 'Negative'.
   - Visualize the distribution of positive and negative ratings using a pie chart.

4. **Exploratory Data Analysis (EDA)**
   - **Review Length Analysis**: Calculate and print the length of a sample review. Create a new column 'Length' to store the length of each review. Visualize the distribution of review lengths using box plots and density distributions.
   - **Word Count Analysis**: Count the number of words in a sample review and print the result. Implement a function to count the number of words in each review and create a new column 'Word_count' to store the counts. Visualize the distribution of word counts using appropriate plots.
   - **Mean Word Length and Sentence Length Analysis**: Calculate the mean word length and mean sentence length for each review. Add 'mean_word_length' and 'mean_sent_length' columns to the DataFrame. Visualize the distributions of these features using appropriate plots.
   - **Text Preprocessing**: Implement a function to preprocess the reviews by converting text to lowercase, removing special characters using regular expressions, and eliminating stopwords using NLTK's 'stopwords' corpus. Update the DataFrame with the cleaned reviews.

5. **Most Frequently Occurring Words Analysis**
   - **Word Frequency Analysis**: Create a corpus by combining all cleaned reviews. Count the frequency of each word using the `Counter` module. Visualize the top 10 most frequently occurring words using a bar plot.
   - **Bigram Frequency Analysis**: Use `CountVectorizer` with a specified n-gram range to generate bigrams from the cleaned reviews. Calculate the frequency of each bigram and create a DataFrame with the results. Visualize the top 10 most frequently occurring bigrams using a bar plot.
   - **Trigram Frequency Analysis**: Similarly, use `CountVectorizer` to generate trigrams from the cleaned reviews. Calculate the frequency of each trigram and create a DataFrame. Visualize the top 10 most frequently occurring trigrams using a bar plot.

6. **Data Cleaning and Preparation**
   - Remove unnecessary columns from the DataFrame, keeping only relevant columns for further analysis. Print the DataFrame's information to verify the changes.

7. **Implementation Considerations**
   - Provide additional information about the implementation, such as downloading NLTK resources (e.g., stopwords) and setting default styling and font size parameters for plots.



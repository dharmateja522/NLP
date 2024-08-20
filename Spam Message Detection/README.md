# Spam Message Detection

## Problem Statement

### Background Information

Spam message detection is crucial for filtering out unwanted messages in various communication platforms. This project focuses on classifying messages into spam or non-spam categories. By analyzing and modeling text data, we can develop effective classifiers that help in automating spam detection.

### Dataset Information

The dataset used in this project contains messages with labels indicating whether they are spam or not. The dataset is typically structured with columns such as:
- **Message**: The text content of the message.
- **Label**: The classification label (e.g., spam or non-spam).

## Approach

1. **Data Loading and Exploration**
   - Load the dataset into a pandas DataFrame.
   - Display the last few rows of the DataFrame to inspect the data.
   - Use `df['source'].unique()` to get unique sources in the dataset.

2. **Model Training and Evaluation**

   For each unique source:
   - **Filter Data**: Filter the DataFrame based on the current source using `df[df['source'] == source]`.
   - **Split Data**: Split the messages and labels into training and testing sets using `train_test_split()`.
   - **Feature Extraction**: Convert text messages into numerical features using `CountVectorizer()`.
   - **Logistic Regression Classifier**:
     - Create and train a logistic regression classifier on the training data.
     - Evaluate its accuracy on the testing data.

3. **Logistic Regression Model**
   - **Feature Extraction**: Use `CountVectorizer()` to convert text messages into a matrix of token counts.
   - **Data Splitting**: Split the data into training and testing sets using `train_test_split()`.
   - **Model Training**: Create an instance of `LogisticRegression()` and fit the classifier on the training data.
   - **Evaluation**: Evaluate the model’s accuracy on the testing data using `score()`.

4. **Neural Network Model**
   - **Tokenization**: Tokenize the text messages using `Tokenizer()` and convert them into sequences of integers.
   - **Data Preparation**: Split the data into training and testing sets and pad sequences to have the same length using `pad_sequences()`.
   - **Model Definition**: Define the neural network model using `Sequential()` and add appropriate layers.
   - **Compilation**: Compile the model with the chosen loss function, optimizer, and evaluation metrics.
   - **Training**: Train the model on the training data and evaluate its accuracy on the testing data.
   - **Visualization**: Visualize training and validation accuracy and loss using `plot_history()`.

5. **Model Evaluation and Comparison**
   - Calculate and print the training and testing accuracies for both the logistic regression and neural network models.
   - Compare the performances of the two models and analyze the results.

6. **Model Tuning (Optional)**
   - **Hyperparameter Tuning**: Define a function `create_model()` that takes hyperparameters as input and returns a compiled neural network model.
   - **Randomized Search**: Specify a hyperparameter grid and perform a randomized search using `RandomizedSearchCV()`.
   - **Evaluation**: Evaluate the model's accuracy on the testing set with the best hyperparameters.
   - **Save Results**: Save and analyze the results to determine the best model configuration.

## Implementation Steps

1. **Import Libraries**
   - Import necessary libraries such as NumPy, Pandas, Scikit-Learn, TensorFlow/Keras, and Matplotlib.

2. **Load and Explore Dataset**
   - Load the dataset and inspect the structure and content.

3. **Preprocess Data**
   - Transform and preprocess data for feature extraction and model training.

4. **Train Models**
   - Implement and train Logistic Regression and Neural Network models.

5. **Evaluate and Compare**
   - Evaluate models and compare their performance.

6. **Tune Models (Optional)**
   - Perform hyperparameter tuning to optimize model performance.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For issues or suggestions, open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to reach out for any questions or feedback!

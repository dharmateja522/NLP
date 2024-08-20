# BBC Articles Summarization

## Problem Statement

### Problem Description

The goal of this project is to develop an algorithm that can automatically summarize BBC articles. Summarization is a natural language processing (NLP) task that generates concise summaries from longer texts while preserving key information. This automation helps users quickly grasp the main points of an article, saving time and effort.

### Background Information

In today's digital era, the volume of information is vast, making it challenging to consume all content efficiently. Automatic summarization techniques leverage machine learning and NLP algorithms to extract important information and create condensed summaries. These techniques are valuable for enhancing content accessibility and usability.

### Dataset Information

We use the [XSum dataset](https://huggingface.co/datasets/xsum), which contains BBC articles paired with single-sentence summaries. The dataset includes:
- **document**: The original BBC article.
- **summary**: The single-sentence summary of the article.
- **id**: Unique identifier for each document-summary pair.

The dataset is divided into training and testing subsets for model evaluation.

## Project Workflow

1. **Importing Libraries**
   - Import necessary libraries such as TensorFlow, Keras, and Transformers.

2. **Loading and Exploring the Dataset**
   - Load the XSum dataset using the `load_dataset` function from the `datasets` library.
   - Inspect the dataset's structure and content.

3. **Preprocessing the Dataset**
   - Add prefixes to documents.
   - Tokenize the input and target sequences.
   - Set maximum input and target lengths.

4. **Creating the Model**
   - Load a pre-trained model checkpoint, such as `t5-small`, using `TFAutoModelForSeq2SeqLM` from the Transformers library.

5. **Configuring the Data Collator**
   - Instantiate a data collator with `DataCollatorForSeq2Seq` for batching tokenized data.

6. **Generating Training and Testing Datasets**
   - Convert tokenized datasets into TensorFlow datasets with appropriate batch sizes.

7. **Compiling and Training the Model**
   - Compile the model with an optimizer and train it on the training dataset.
   - Use the RougeL metric to evaluate performance during training.

8. **Evaluating the Model**
   - Assess the trained model on the testing dataset using the RougeL metric.

9. **Summarizing BBC Articles**
   - Utilize the trained model to generate summaries for new BBC articles.
   - Set up a summarization pipeline and generate summaries with specified lengths.

## Framework

1. **Install Required Libraries**
   - Install necessary libraries using pip:
     ```bash
     pip install tensorflow transformers datasets nltk
     ```

2. **Import Libraries**
   - Import required libraries in your code:
     ```python
     import tensorflow as tf
     from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, pipeline
     from datasets import load_dataset
     import nltk
     ```

3. **Set Constants**
   - Define constants such as:
     ```python
     TRAIN_TEST_SPLIT = 0.1
     MAX_INPUT_LENGTH = 512
     MIN_TARGET_LENGTH = 10
     MAX_TARGET_LENGTH = 50
     BATCH_SIZE = 16
     LEARNING_RATE = 5e-5
     MAX_EPOCHS = 3
     ```

4. **Load and Explore Dataset**
   - Load the dataset and print its structure:
     ```python
     dataset = load_dataset("xsum")
     print(dataset)
     ```

5. **Preprocess Dataset**
   - Create and apply a preprocessing function:
     ```python
     def preprocess_function(examples):
         # Add prefix, tokenize, and set lengths
         pass

     tokenized_datasets = dataset.map(preprocess_function, batched=True)
     ```

6. **Tokenize and Map Dataset**
   - Tokenize the dataset:
     ```python
     tokenized_datasets = tokenized_datasets.map(preprocess_function, batched=True)
     ```

7. **Load Pretrained Model**
   - Load the model:
     ```python
     model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")
     ```

8. **Configure Data Collator**
   - Set up the data collator:
     ```python
     data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
     ```

9. **Create Training and Testing Datasets**
   - Convert to TensorFlow datasets:
     ```python
     train_dataset = tokenized_datasets["train"].to_tf_dataset(
         columns=["input_ids", "attention_mask", "labels"],
         batch_size=BATCH_SIZE,
         shuffle=True,
         collate_fn=data_collator
     )
     ```

10. **Compile and Train the Model**
    - Compile and train:
      ```python
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
      model.fit(train_dataset, epochs=MAX_EPOCHS, validation_data=tokenized_datasets["validation"])
      ```

11. **Evaluate the Model**
    - Evaluate performance:
      ```python
      from datasets import load_metric
      rouge = load_metric("rouge")
      results = model.evaluate(test_dataset)
      ```

12. **Summarize BBC Articles**
    - Summarize articles:
      ```python
      summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
      summary = summarizer("Your BBC article text here", min_length=MIN_TARGET_LENGTH, max_length=MAX_TARGET_LENGTH)
      print(summary)
      ```


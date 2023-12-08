# Sentimental-Analysis

Sentiment Analysis using RNN Documentation

This documentation outlines the implementation of Sentiment Analysis using a Recurrent Neural Network (RNN) for a dataset containing movie reviews. The code is executed on Google Colaboratory and involves data loading, preprocessing, model construction, training, and evaluation.

Data Loading and Preprocessing

Mount Google Drive:
Google Drive is mounted to access the dataset file.

Load Dataset:
The dataset, containing reviews and sentiments (positive/negative), is loaded from a CSV file using pandas.

Train-Test Split:
The dataset is split into training and testing sets, maintaining a 80-20 ratio.

Text Preprocessing:
The reviews undergo preprocessing, including lowercase conversion, tokenization, lemmatization, and removal of stopwords.

Vocabulary Creation:
A vocabulary is created, mapping each unique word to a numerical ID.

Text to Numerical Conversion:
Reviews are converted to numerical sequences based on the vocabulary.

Padding Sequences:
Sequences are normalized by padding to a specified maximum length (150).

Data Loading for PyTorch:
The preprocessed data is transformed into PyTorch TensorDataset and loaded into batches using DataLoader.
RNN Model Construction

Define RNN Model:
An RNN model is defined with an embedding layer, LSTM unit, dropout layer, and linear layer with a sigmoid output.

model = RNN_SA(num_layers, hidden_layer_size, embedding_layer_size, output_layer_size, vocabulary_size, dropout_probability=0.3)


Training the RNN Model

Training Configuration:
Hyperparameters like learning rate (1e-3), number of epochs (3), BCE loss, and Adam optimizer are configured.

Training Loop:
The model is trained for multiple epochs, with progress tracked using tqdm. The training loop involves forward pass, backward pass, loss calculation, and parameter updates.

Model Saving:
The model's state is saved when the validation loss decreases, ensuring the best-performing model is retained.
Evaluation and Visualization

Performance Evaluation:
The model is evaluated on the test set, and accuracy and loss metrics are calculated.
Visualization:

Training and testing accuracy, as well as training and testing loss, are visualized using matplotlib.
Results

Results Summary:
The model achieves reasonable accuracy on sentiment prediction, with training and testing metrics visualized for analysis.

Conclusion:
This implementation demonstrates the process of sentiment analysis using an RNN, showcasing the model's learning and performance on movie reviews. The code provides a foundation for understanding and extending sentiment analysis tasks using deep learning techniques.

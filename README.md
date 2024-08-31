# AI Chatbot with Deep Learning Using PyTorch and Gradio



## Overview

This project implements an AI chatbot using deep learning techniques, specifically leveraging PyTorch for model development and Gradio for creating a user-friendly web interface. The chatbot is designed to understand user input, identify intent, and provide appropriate responses based on predefined intents and patterns. This project showcases the use of natural language processing (NLP) techniques to create an intelligent conversational agent.

## Features

- **Deep Learning Model**: A fully connected neural network built with PyTorch, trained on a custom dataset of intents.
- **Natural Language Processing (NLP)**: Utilizes tokenization, stemming, and bag-of-words model to process and understand user inputs.
- **User-Friendly Interface**: A web-based interface created with Gradio to easily interact with the chatbot.
- **Dynamic Response Generation**: The chatbot provides responses based on the predicted intent with a confidence threshold to handle unknown inputs gracefully.


To run this project locally, follow these steps:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/chatbot.git
    cd chatbot
    ```

2. **Install the Required Packages**:

    Install all the dependencies listed in the `requirements.txt` file using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download NLTK Data**:

    Ensure you have the required NLTK packages by running:

    ```python
    import nltk
    nltk.download('punkt')
    ```

4. **Run the Chatbot**:

    Launch the chatbot using Gradio by running the following command:

    ```bash
    python chatbot.py
    ```

    This will start a local server and provide a URL to interact with the chatbot via a web browser.

## How It Works

1. **Data Preparation**: The `intents.json` file contains multiple intents, each having several patterns (sample sentences) and corresponding responses. This data is tokenized, stemmed, and converted into a numerical format (bag-of-words) for training.
   
2. **Model Training**: A PyTorch neural network is trained using the processed data to classify input sentences into one of the predefined intents. The model learns to associate different patterns with their corresponding tags or intents.
   
3. **Inference**: During interaction, the user input is tokenized, stemmed, and transformed into a bag-of-words format. This input is then passed through the trained model to predict the most likely intent. Based on the intent and the model's confidence, the chatbot generates an appropriate response or indicates that it didn't understand the input.

4. **User Interface**: The Gradio library provides a simple web interface where users can enter their queries, and the chatbot responds in real-time.

## Customization

To customize the chatbot:

1. **Modify Intents**: Update the `intents.json` file with your own intents, patterns, and responses.
2. **Retrain the Model**: After modifying the intents, retrain the model by running the training script.
3. **Adjust Model Parameters**: Modify hyperparameters such as learning rate, batch size, and epochs in the training script to optimize performance.

## Technologies Used

- **Python**: Main programming language.
- **PyTorch**: Deep learning framework used to build and train the neural network.
- **NLTK**: Natural Language Toolkit for text processing, tokenization, and stemming.
- **Gradio**: Library to create an interactive user interface for the chatbot.
- **JSON**: Used for structuring training data.

## Future Enhancements

- Integrate more sophisticated NLP techniques like Named Entity Recognition (NER) and Parts of Speech (POS) tagging.
- Implement a more advanced model architecture, such as Transformers or BERT.
- Expand the dataset to include more intents and improve the chatbot's conversational ability.
- Deploy the chatbot as a web app using a cloud platform like AWS or Heroku.

## Contributing

Contributions are welcome! If you have any suggestions, feel free to create an issue or submit a pull request.

## Acknowledgments

- Special thanks to the creators of [Gradio](https://gradio.app/) and [PyTorch](https://pytorch.org/).
- The training data was inspired by various open-source chatbot projects.


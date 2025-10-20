# Image-Caption-Generator-using-CNN-and-LSTM
• Built an Image Caption Generator using InceptionV3 and LSTM to generate natural language captions for 8,000 images from the Flickr8k dataset. • Implemented Greedy and Beam Search decoding with BLEU score evaluation, achieving BLEU-1 score of 0.71 and BLEU-4 score of 0.54, integrating Computer Vision and NLP for contextual caption generation.

1. Objective

To develop a deep learning model capable of understanding the visual content of an image and generating a coherent, human-like description.

2. Technologies Used

Libraries: TensorFlow, Keras, NumPy, Pandas, Seaborn, PIL, Scikit-learn, Matplotlib

Models: InceptionV3 (for image feature extraction), LSTM (for text sequence generation)

Evaluation: BLEU Score (for caption quality assessment)

3. Methodology
Step 1: Data Preparation

Load and preprocess the Flickr8K dataset containing images and corresponding captions.

Clean captions by removing punctuation, digits, and special symbols.

Add start and end tokens to define caption boundaries.

Split the dataset into training (70%), validation (15%), and testing (15%) sets.

Step 2: Feature Extraction

Use InceptionV3, a pre-trained CNN on ImageNet, to extract 2048-dimensional feature vectors from each image.

Remove the classification layer to use the model as a feature extractor.

Step 3: Text Tokenization

Use Keras Tokenizer to convert cleaned captions into integer sequences.

Build a vocabulary of 8,586 unique words.

Step 4: Data Generator

Implement a generator function to dynamically load batches of image features and captions, improving memory efficiency during training.

Step 5: Model Architecture

Image Encoder: Dense and BatchNormalization layers applied to image features.

Caption Decoder: Embedding and LSTM layers to process text sequences.

Fusion Layer: Combine outputs from CNN and LSTM using add() operation.

Output Layer: Dense layer with Softmax activation for next-word prediction.

Step 6: Model Training

Trained using Adam optimizer and categorical cross-entropy loss.

Implemented EarlyStopping and Learning Rate Scheduling to avoid overfitting.

Trained for 15 epochs on the processed dataset.

Step 7: Evaluation & Decoding

Two decoding strategies implemented:

Greedy Search: Selects the most probable word at each step.

Beam Search: Explores multiple best candidate sequences for improved accuracy.

Evaluated using BLEU-1 and BLEU-2 scores to measure caption quality.

4. Results

The model successfully generated meaningful captions such as:

“A man is riding a skateboard on a ramp.”

“Two dogs are running through the grass.”

Beam Search generated more contextually accurate and fluent captions than Greedy Search.

Visualization showed good alignment between image content and generated text.

5. Key Findings

Transfer Learning (using InceptionV3) significantly improved training efficiency.

LSTM Decoder effectively captured word dependencies for fluent caption generation.

Beam Search outperformed Greedy Search in BLEU scores, producing more natural sentences.

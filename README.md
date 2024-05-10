# Sign-Language-Detection-using-CNN-Architecture
## Abstract
Sign language is a language used as a manual communication method used by people who are deaf, mute, etc. Hand gesture is one of the main methods used in this language for non-verbal communication. It is the primary method of communication for a huge amount of people worldwide. Artificial intelligence is booming in the fields of object recognition, Machine Translation, Computer Vision, etc. Artificial Intelligence can be used to detect and recognize Sign Language via various techniques, such as image processing, motion detection, and gesture recognition by applying different popular machine learning and deep learning algorithms. In this project, we will be using the ISL (Indian Sign Language) dataset from Kaggle as well as ISLTranslate, which is a dataset containing frames and videos containing sign language, visual language, fingerspelling, and facial expressions in Indian Sign Language. The model will use a Deep Learning architecture that is efficient in Image recognition (Convolutional Neural Network Architecture). Using this model, we will train the model to recognize hand gestures and movement of hands with the dataset acquired. Once the model can successfully classify and recognize the images in real time, it will generate English text according to Sign Language, which will make communication with mute and deaf people easy.

**Keywords: -** mute, artificial intelligence (AI), object recognition, machine translation, computer vision, image processing, motion detection, gesture recognition, machine learning, deep learning, Kaggle, ISLTranslate dataset, Convolutional Neural Network (CNN), real-time classification.

## Problem identification
Identifying problems and challenges in the task of recognizing Indian Sign Language using a CNN (Convolutional Neural Network) is an essential step in building an effective and reliable model. Here are some key problem areas to consider:
1. **Data Quality and Quantity:**
  - Data Augmentation can be used to increase and achieve a higher accuracy.
  - Label Accuracy: Ensuring the accuracy of labels in the ISL Dataset is critical, as mislabelling or errors in the alphanumeric sign can lead to incorrect model training and bad accuracy.
  - Data Variability in Gesture: Sign Language includes a variety of gestures as well as expression making the CNN model harder to accurately detection and classify the language. 
2. **Model Complexity and Performance:**
  - Complexity: Using the appropriate level of complexity for the CNN model is important as it may lead to overfitting, especially when we are dealing with limited data. However, very simple models may tend to underfit. Balancing the model complexity to acquire an optimal accuracy is a key aspect.
  - Generalization: The CNN model needs to generalize well to unseen data, including variations in image quality, lighting conditions, Noisy as well as dull image data, which may help it to classify the gestures precisely in real time.
  - Model Interpretability: Understanding how the CNN architecture arrives at its decision is a practical concern as for its deployment for system with low/limited resources. Developing interpretable models and explainable model predictions is a crucial in the context of Indian Sign Language recognition.
3. **Dynamic Gestures of Sign Language:**
  - Motion speed and duration of gesture: Various gestures vary in speed and duration, which adds the complexity to the detection and classification process. The CNN architecture needs to be capable as well as be trained so that it can accurately classify the gesture which are moving in different speed.
  - Gesture Transition: Transition between gestures is crucial to the nature of communicating sign language. Smoother detection of transition is a key aspect for understanding the meaning of series of sign gestures.
4. **Hardware and Computational Resources:**
  - Model Size: The CNN model may require substantial computational resources, including GPUs, for training. This can be a limitation for smaller institutions or research groups.
  - Inference Efficiency: Deploying a computationally intensive model for real-time for motion gesture communication with high accuracy may not be possible. Optimizing the model for efficient inference is important.
5. **Real-Time Processing Requirement:**
  - Model Size: Very large CNN models with many parameters will not be suitable for real-time processing/identifying sign language on devices with less resources. Reducing the model size may impact the performance of the model.
  - Latency: The delay/latency between the input and output matrix, is an important factor for real-time applications. High-latency may prove to suitable for applications where timely classification of the sign language is required. Optimizing the architecture without impacting the performance while reducing the latency.
## Proposed system design
The project integrates the Rectified Linear Unit (ReLU) activation functions with a self-trained Convolutional Neural Network (CNN) model, the proposed system design for hand gesture detection leverages cutting-edge machine learning techniques to provide reliable gesture recognition. An outline of the system design is provided here:
  - Data Acquisition and Dataset Selection: The system makes use of the ISL Translate dataset, which is a varied collection of pictures that correspond to different Indian Sign Language (ISL) hand gesture characters. For training and validation, each hand motion character requires an average of 1200 photos.
  - Preprocessing and Model Training: The dataset is preprocessed to normalise pixel values, improve contrast, and standardise image sizes before the model is trained. Next, using the preprocessed dataset, the self-trained CNN model is trained to identify complex patterns and characteristics connected to various hand motions.
  - Model Architecture and Activation Functions: To extract hierarchical features from input images and carry out classification tasks, the CNN architecture consists of several convolutional layers, pooling layers, and fully connected layers. To add non-linearity and improve the model's capacity to learn complicated representations, ReLU activation functions are utilised. Softmax function is used for predicting the class of the input image or frame.
  - Real-Time Hand Gesture Recognition: By using the trained CNN model, hand movements recorded by a camera or depth sensor may be instantly interpreted in real-time. As inbound picture frames are processed, the system determines which hand gesture character goes with them and gives the user instant feedback.
  - User Interface Integration: The system includes an intuitive user interface that shows users' hand motions in real time and provides them with visual feedback. In applications like interactive interfaces, virtual reality settings, and assistive technologies, this interface improves user involvement and makes communication more fluid.

       ![image](https://github.com/SomyanshAvasthi/Sign-Language-Detection-using-CNN-Architecture/assets/107310391/abfa7609-fa4a-4037-8624-129ce92c18e4)
## Result and discussion
The ISL Translate dataset was used to train the CNN model extensively, teaching it discriminative features through photos of hand movements. By training the model to identify minute differences and nuances in hand gestures, with an average of 1200 photos per hand gesture character, robust classification performance was ensured. Recall and confusion matrix analysis were two validation strategies used during training to evaluate the model's accuracy and performance characteristics.

![image](https://github.com/SomyanshAvasthi/Sign-Language-Detection-using-CNN-Architecture/assets/107310391/1ec1482e-41a7-4875-bf8b-a0d0ce477a09)                   ![image](https://github.com/SomyanshAvasthi/Sign-Language-Detection-using-CNN-Architecture/assets/107310391/8bc84bcf-9cff-4d52-ab1e-fcd8f8cd5201)                    ![image](https://github.com/SomyanshAvasthi/Sign-Language-Detection-using-CNN-Architecture/assets/107310391/1bdde91d-aa98-4317-81c1-4418da94b77d)

  _(Fig 1 - Sample of results obtained)_
  
![image](https://github.com/SomyanshAvasthi/Sign-Language-Detection-using-CNN-Architecture/assets/107310391/fbbb3683-beb3-4c8c-85dc-7743bacdcf23)
                                                  
  _(Fig 2 - Confusion Matrix with Labels)_

![image](https://github.com/SomyanshAvasthi/Sign-Language-Detection-using-CNN-Architecture/assets/107310391/267c32dc-674b-4cec-ab9d-a2360c547eb9)

   _(Fig 3 - Training Accuracy Graph)_
  
![image](https://github.com/SomyanshAvasthi/Sign-Language-Detection-using-CNN-Architecture/assets/107310391/566cd433-26a3-429a-9a28-37ef96315d5f)

   _(Fig 4 - Training Loss function Graph)_


   **Results obtained are mentioned below :-**
   
   ➔	***Accuracy Attained - 0.99111008644104***
   
   ➔	_**Loss - 0.06643851101398468**_
   
   ➔	***F1 Score - 0.98*** 
   
   ➔	***Precision - 0.99***      
   
   ➔	***Recall - 0.97***

## Conclusion
In order to sum up, our project represents a major advancement in the field of hand gesture detection technology. Using the vast ISL Translate dataset and a self-trained CNN model with ReLU activation functions, we have obtained impressive accuracy, which has been confirmed by thorough recall and confusion matrix analysis. Our technology promises new applications in assistive technologies and interactive interfaces, providing users with smooth and natural interactions thanks to its real-time detection capabilities. Our project opens the door for future innovation and advances gesture recognition technology to unprecedented levels of usefulness and accessibility for a wide range of user demographics.

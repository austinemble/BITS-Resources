# Birla Institute of Technology & Science, Pilani
### Work Integrated Learning Programmes Division
### Digital Learning Handout

---

## Part A: Content Design

### Course Information
| Parameter | Details |
| :--- | :--- |
| **Course Title** | Deep Neural Networks |
| **Course No(s)** | AIML * ZG511 |
| **Credit Units** | 4 |
| **Credit Model** | 3-1-0 |
| **Course Author** | Prof. Seetha Parameswaran |
| **Instructors** | Eht E Sham (Lead), Nivethitha S, Sankara Nayaki K, Anupama Angadi, Chandra Prakash Manglani, Charu Kathuria, Shakshi Sharma |
| **Version No** | 2 |
| **Date** | 20/09/2025 |

---

### Course Description
Introduction to neural networks, linear neural networks for regression and classification, deep feed forward neural networks, back propagation, deep network training, regularisation and optimisation, transfer learning and fine-tuning, convolution neural networks, recurrent neural networks, attention models, transformers, and vision transformers.

---

### Course Objectives
| Code | Course Objective |
| :--- | :--- |
| **CO1** | Introduce the fundamental concepts of neural networks, including biological inspiration, perceptrons, linear models, and deep feedforward networks. |
| **CO2** | Familiarize students with convolutional and recurrent neural network architectures and their application to vision and sequence tasks. |
| **CO3** | Examine attention mechanisms and transformer architectures, including vision transformers and large-scale pretraining paradigms. |
| **CO4** | Provide an understanding of optimisation and regularisation techniques essential for training deep neural networks effectively. |

---

### Text Book(s)
| Code | Reference / Citation |
| :--- | :--- |
| **T1** | Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into deep learning*. Cambridge University Press. |

---

### Reference Book(s) & Other Resources
| Code | Reference / Citation |
| :--- | :--- |
| **R1** | Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). *Deep learning* (Vol. 1, No. 2). Cambridge: MIT press. |
| **R2** | Chollet, F., & Chollet, F. (2021). *Deep learning with Python*. Manning. |
| **R3** | Jurafsky, D., & Martin, J. H. (2026). *Speech and language processing* (3rd ed. draft). |

---

### Learning Outcomes
Students will be able to:

| Code | Learning Outcome |
| :--- | :--- |
| **LO1** | Implement linear and deep feedforward neural networks from scratch, applying forward propagation, backpropagation, and gradient-based weight updates. |
| **LO2** | Design and implement CNN architectures for image classification tasks, and apply transfer learning for domain adaptation. |
| **LO3** | Implement RNN, LSTM, and GRU architectures for sequence modelling tasks, and analyse the role of gating mechanisms in addressing vanishing gradients. |
| **LO4** | Implement attention mechanisms and transformer architectures, and evaluate their application to NLP and vision tasks including ViT. |
| **LO5** | Apply optimisation algorithms and regularisation techniques to improve deep network training. |

---

### Modular Content Structure

1. **Introduction to Deep Neural Network**
   - Deep learning definition, motivation, and key components
   - Supervised learning examples and wake word detection

2. **Artificial Neuron and Perceptron**
   - Artificial Neuron vs Biological Neuron
   - Perceptron and Perceptron learning algorithm

3. **Linear Neural Network for Regression and Classification**
   - Single neuron in the output layer for regression
   - Single neuron in the output layer for classification
   - Multiple neurons in the output layer for classification

4. **Deep Feedforward Neural Networks or Multi-Layer Perceptrons**
   - Non-linearity and hidden layers
   - MLP architecture
   - Forward and backward propagation and weight updates
   - Impact of depth and width in DFNN

5. **Convolutional Neural Networks**
   - Convolution operation; feature maps, padding, stride, pooling
   - Design of CNN
   - LeNet; deep CNN architectures: VGG, ResNet
   - Transfer learning and fine-tuning

6. **Recurrent Neural Networks**
   - RNN with hidden states; backpropagation through time
   - LSTM, GRU; stacked and bidirectional RNNs
   - Encoder-decoder architecture

7. **Attention Mechanism**
   - QKV concept
   - Scaled dot-product attention
   - Self attention and multi-head attention

8. **Transformer**
   - Transformer architecture: encoder, decoder, positional encoding
   - Vision transformers; large-scale pretraining: BERT, GPT

9. **Optimization of Deep models**
   - Gradient descent variants: SGD, momentum, Adagrad, RMSProp, Adam
   - Optimization challenges: saddle points, vanishing gradients

10. **Regularization for Deep models**
    - Overfitting, underfitting, model selection
    - Weight decay, dropout, batch normalisation, layer normalisation

---

## Part B: Learning Plan

### Contact Sessions & Schedule

| Contact Session | List of Topic Title | Sub-Topics | Reference |
| :---: | :--- | :--- | :--- |
| **1** | Introduction and Motivating Example | Deep Learning - definition<br>Understanding how deep learning works<br>Why Deep Learning?<br>Motivating Example of Wake Word Detection<br>Key components - Data, Models, Objective function, Optimization Algorithms<br>Supervised learning examples | T1 - Ch 1<br>R3 - Ch 1.1.1, 1.1.2, 1.1.5 |
| **2** | Artificial Neuron and Perceptron | Biological neuron vs Artificial Neuron<br>Connectionism model<br>Perceptron<br>Perceptron learning algorithm for NOT, AND, OR gates<br>Perceptron for linearly separable data<br>XOR Problem - demonstrate perceptron fails to learn - non-linearly separable data<br>**Demo:** Implement AND, OR, and XOR gates using a Perceptron from scratch; demonstrate that Perceptron fails to learn XOR. | Class Notes |
| **3** | Linear Neural Networks for Regression | Single neuron for regression - how much or how many<br>Data, linear model with a single neuron and a squared loss function for a single neuron for regression (no hidden layers)<br>Training using the batch gradient descent algorithm and Prediction or inference<br>**Demo:** Implement single neuron regression from scratch using batch gradient descent; train on Auto MPG (UCI) dataset; plot loss curve and predictions | T1 – Ch 3<br>T1 - Ch 12 |
| **4** | Linear Neural Networks for Classification | Single neuron for classification - which category<br>Data, linear model with a single neuron, sigmoid activation function, and binary cross-entropy loss for a single neuron for classification (no hidden layers)<br>Training using the stochastic gradient descent algorithm and Prediction or inference<br>Multi-class classification - which category<br>Data, linear model with multiple output neurons, softmax activation function, and cross-entropy loss for multiple neurons for classification (no hidden layers)<br>Training using the mini-batch stochastic gradient descent algorithm and Prediction or inference<br>**Demo:** Implement single neuron binary classification from scratch using sigmoid activation and binary cross-entropy loss; train on Breast Cancer Wisconsin (UCI) dataset; implement multi-class classification with softmax and cross-entropy loss; train on Iris (UCI) dataset | T1 – Ch 4<br>T1 - Ch 12 |
| **5** | Deep Feedforward Neural Networks (DFNN) or Multi Layer Perceptron (MLP) for Classification | MLP as a solution for XOR<br>Hidden layers and Non-linearity<br>Forward Propagation (use vectorization and algorithm for forward pass)<br>Define loss functions and compute the error<br>Backward Propagation (use vectorization and algorithm for backprop)<br>Weight updation using gradients (use vectorization and algorithm for backprop)<br>Impact of depth and width in DFNN<br>**Demo:** Implement MLP forward propagation and backpropagation from scratch using vectorization; demonstrate XOR solution with hidden layer; show impact of depth and width on decision boundary | T1 – Ch 5<br>R4 - Ch 7.3, 7.5 |
| **6** | Convolutional Neural Networks | Image Data<br>Invariance, Locality, and Channels<br>Convolutions operation<br>Learning a Kernel<br>Feature map and receptive field<br>Padding and Stride<br>1x1 convolution<br>Pooling<br>Multiple channels<br>Putting it all together in LeNet<br>**Demo:** Build and train a custom CNN on an image dataset; demonstrate feature maps and receptive fields | T1 – Ch 7<br>R3 - 2.2.11 |
| **7** | Deep Convolutional Neural Networks | Networks Using Blocks (VGG) - VGG Blocks<br>Network in Network (NiN) - NiN Blocks and NiN Model<br>GoogLeNet - Inception Blocks<br>Residual Networks (ResNet) - Residual Blocks | T1 – Ch 8 |
| **8** | Deep Convolutional Neural Networks | Transfer Learning<br>Fine-tuning for transfer learning<br>**Demo:** Fine-tune a pretrained VGG16 for a domain image classification task | Class notes |
| **9** | Recurrent Neural Networks | Raw Text into Sequence Data<br>Recurrent connection<br>Recurrent Neural Networks with Hidden States<br>Backpropagation Through Time<br>RNNs for NLP tasks<br>Encoder–Decoder Architecture<br>Teacher Forcing<br>Loss Function with Masking<br>Training and Prediction<br>**Demo:** Build and train an RNN for a time series prediction task; demonstrate hidden state evolution and backpropagation through time | T1 - Ch 9<br>R4 – Ch 9<br>R3 - 2.2.10 |
| **10** | Deep Recurrent Neural Networks | Long Short-Term Memory (LSTM)<br>Gated Memory Cell<br>Gated Recurrent Units (GRU)<br>Reset Gate and Update Gate, Candidate Hidden State<br>Stacked RNN architectures<br>Bidirectional Recurrent Neural Networks<br>**Demo:** Implement LSTM and GRU; compare RNN vs LSTM vs GRU on sequence task | T1 – Ch 10<br>R4 – Ch 9 |
| **11** | Attention Mechanism | Queries, Keys, and Values<br>Attention Scoring Functions<br>Dot Product Attention<br>Scaled Dot Product Attention<br>Bahdanau Attention Mechanism<br>Multi-Head Attention<br>Self-Attention, Cross Attention<br>Positional Encoding | T1 – Ch 11<br>R4 - Ch 9 |
| **12** | Transformer | Transformer architecture<br>Model, Positionwise Feed-Forward Networks, Residual Connection, and Layer Normalization<br>Encoder and Decoder Transformer block<br>Encoder-Only, Encoder–Decoder, Decoder-Only<br>Residual view for transformer<br>**Demo:** Build and train a Transformer for a time series task; visualize attention maps | T1 – Ch 11<br>R4 – Ch 10 |
| **13** | Transformer | Transformers for Vision<br>Model, Patch Embedding, Vision Transformer Encoder, Training and Evaluation<br>Large-Scale Pretraining with Transformers<br>Scalability | T1 – Ch 11<br>R4 - Ch 10.7 |
| **14** | Optimization of Deep models | Goal of Optimization<br>Optimization Challenges in Deep Learning<br>Review of Gradient Descent, Stochastic Gradient Descent and Minibatch Stochastic Gradient Descent (already learned in M3, M4)<br>Momentum<br>Adagrad Algorithm<br>RMSProp Algorithm<br>Adadelta Algorithm<br>Adam Algorithm<br>**Demo:** Implement and compare SGD, momentum, Adagrad, RMSProp, Adam on a common deep network; plot loss curves and convergence speed for each optimizer | T1 – Ch 12 |
| **15** | Regularization for Deep models | Generalization for regression and classification<br>Training Error and Generalization Error<br>Underfitting or Overfitting<br>Model Selection<br>Weight Decay and Norms<br>Environment and Distribution Shift<br>Generalization in Deep Learning<br>Dropout<br>Batch Normalization<br>Layer Normalization<br>**Demo:** Demonstrate overfitting on a deep network; apply dropout, batch normalization, layer normalization, and weight decay; compare training vs validation loss before and after regularization | T1 – Ch 3.6, 3.7<br>T1 - Ch 4.6, 4.7<br>T1 - Ch 5.5, 5.6<br>T1 - Ch 8.5<br>T1 - Ch 11.7 |
| **16** | Review | Review of contact sessions 1 to 15 | |

---

### Experiential Learning Components
- **Lab work:** 10
- **Project work:** 0
- **Case Study:** 0
- **Simulation:** 0
- **Work Integrated Learning Assignment:** 2 Assignments
- **Design work / Field work:** 0

#### Objective of Experiential Learning Component
Hands-on sessions on the implementation of:
- Develop practical skills in implementing linear neural networks for regression and classification tasks from scratch
- Build hands-on experience with deep feedforward neural networks including forward propagation, backpropagation, and gradient-based weight updates
- Gain applied understanding of convolutional neural network architectures for image recognition, including deep CNN variants and transfer learning
- Implement and evaluate recurrent neural network architectures — RNN, LSTM, GRU — for sequence modelling and time series prediction
- Apply attention mechanisms and multi-head attention through hands-on implementation
- Build and train transformer architectures for time series and vision tasks, and fine-tune pretrained vision transformers
- Compare and analyse optimisation algorithms — SGD, momentum, Adagrad, RMSProp, Adadelta, Adam — through implementation and convergence analysis
- Apply and evaluate regularisation techniques — dropout, batch normalisation, layer normalisation, weight decay — to control overfitting in deep networks

#### Scope of Experiential Learning Component
- **Programming language:** Python
- **Tools and libraries:** Jupyter, Numpy, Pandas, ScikitLearn, Matplotlib, Tensorflow, Pytorch, Keras
- **Development Environment:** Python 3.8+, TensorFlow 2.x / PyTorch 2.0+
- **Lab Infrastructure:** WILP Virtual Lab

#### List of Experiments
| Lab No | Lab Objective | Session Reference |
| :---: | :--- | :---: |
| **1** | Building a linear neural network for regression: Auto MPG prediction with a single neuron implementation from scratch | 3 |
| **2** | Implementing a linear neural network for binary classification: Breast Cancer Wisconsin prediction with a single neuron from scratch | 4 |
| **3** | Building multi-class classification with multiple output neurons: Iris dataset prediction from scratch | 4 |
| **4** | Implementing a Deep Feedforward Neural Network (MLP) with forward and backward propagation from scratch | 5 |
| **5** | Building a Convolutional Neural Network (CNN) with LeNet architecture for image classification | 6 |
| **6** | Implementing deep CNN architectures (AlexNet/VGG/ResNet) and transfer learning for image recognition | 7, 8 |
| **7** | Building Recurrent Neural Networks (RNN, LSTM, GRU) for sequence modeling and time series prediction | 9, 10 |
| **8** | Implementing attention mechanisms and the Transformer architecture for time series and vision tasks | 11, 12, 13 |
| **9** | Comparing optimization algorithms (SGD, Momentum, Adam, RMSProp, Adagrad, Adadelta) with implementation from scratch | 14 |
| **10** | Implementing and comparing regularization techniques (dropout, batch normalization, layer normalization, weight decay) | 15 |

---

### Evaluation Scheme

| Evaluation Component | Name | Type | Weight | Duration | Day, Date, Session, Time |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **EC – 1\*** | Quiz | Online | 10% | 1 week | August 10-20, 2026 |
| **EC – 1\*** | Assignment | Online | 20% | 10 days | August 27-7 Sept 2026 |
| **EC - 2** | Mid-Semester Test | Closed Book | 30% | 2 hours | 19/09/2026 (FN) |
| **EC - 3** | Comprehensive Exam | Open Book | 40% | 2 ½ Hours | 05/12/2026 (FN) |

*Legend: EC = Evaluation Component; AN = After Noon Session; FN = Fore Noon Session*

- **Syllabus for Mid-Semester Test (Closed Book):** Topics in Contact sessions 1 to 8.
- **Syllabus for Comprehensive Exam (Open Book):** All topics.

---

### Important Links and Information
- **eLearn Portal:** [https://elearn.bits-pilani.ac.in](https://elearn.bits-pilani.ac.in)
- Students must visit the eLearn portal regularly and stay updated with the latest announcements and deadlines.
- **Contact Sessions:** Students should attend the online lectures as per the schedule provided on the eLearn portal.

### Evaluation Guidelines
- EC-1 consists of either two Assignments or three Quizzes. Students will attempt them through the course pages on the eLearn portal. Announcements will be made on the portal in a timely manner.
- **For Closed Book tests:** No books or reference material of any kind will be permitted.
- **For Open Book exams:** “open book” means text/ reference books (publisher copy only) and does not include any other learning material. No other learning material will be permitted during the open book examinations. For Detailed Guidelines refer to the attached document.
- **Make-Up Policy:** If a student is unable to appear for the Regular Test/Exam due to genuine exigencies, the student should follow the procedure to apply for the Make-Up Test/Exam, which will be made available on the eLearn portal. The Make-Up Test/Exam will be conducted only at selected exam centres on the dates to be announced later.
- It shall be the responsibility of the individual student to be regular in maintaining the self-study schedule as given in the course handout, attend the online lectures, and take all the prescribed evaluation components such as Assignments/Quizzes, Mid-Semester Tests and Comprehensive Exams according to the evaluation scheme provided in the handout.

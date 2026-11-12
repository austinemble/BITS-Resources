# Deep Neural Networks — 50 Questions Examination Bank

UNIVERSITY EXAMINATION – QUESTION BANK

Course: Deep Neural Networks

50 Questions × 6 Marks each  •  Total: 300 Marks

Instructions: Each question carries 6 marks. Difficulty is indicated against each question — Easy (15 questions), Medium (25 questions), Tough (10 questions). Show all steps for numerical questions. Notation follows the course slides.

## Module 1: Introduction to Deep Learning

### Q1. [Easy] Define Deep Learning. Explain where Deep Learning sits within the broader fields of Artificial Intelligence and Machine Learning, and state what the word “deep” refers to in deep learning.  (6 Marks)

### Q2. [Easy] List and briefly explain the four core components of any deep learning problem. Support your answer with the wake-word detection example.  (6 Marks)

### Q3. [Medium] “Scale drives deep learning progress” (Andrew Ng). Explain this statement with the performance vs. amount-of-labelled-data curves for traditional ML and small/medium/large neural networks. Also list any four additional reasons for the popularity of deep learning.  (6 Marks)

### Q4. [Medium] Differentiate between supervised, unsupervised and reinforcement learning, giving one example of each. Under supervised learning, list any four kinds of tasks.  (6 Marks)

## Module 2: Artificial Neural Networks and the Perceptron

### Q5. [Easy] Describe the structure of a biological neuron and compare it with an artificial neuron on any four points.  (6 Marks)

### Q6. [Easy] Define the perceptron. Who introduced it and when? Write the perceptron computation and its output (decision) rule, and draw/describe its structure with bias.  (6 Marks)

### Q7. [Medium] Show how a single perceptron can represent the two-input AND gate. Derive the four inequality constraints on the parameters w₀ (bias), w₁, w₂ from the truth table, and give one valid parameter set. Sketch/describe the linear separation in the x₁–x₂ plane.  (6 Marks)

### Q8. [Medium] State the perceptron learning algorithm, clearly writing the weight and bias update rules. State the perceptron convergence theorem.  (6 Marks)

### Q9. [Tough] Using the perceptron learning algorithm with initial weights w₀ = w₁ = 0, learning rate η = 1 and output rule ŷ = 1 if h ≥ 0 else 0, trace the training of a perceptron for the NOT gate (examples: x₁=0, t=1 and x₁=1, t=0) epoch by epoch until convergence. Report the final weights.  (6 Marks)

### Q10. [Medium] What is the XOR problem? Using the XOR truth table and a sketch of the four points in the x₁–x₂ plane, explain why a single perceptron cannot solve it. Define linearly separable data and state the solution to the XOR problem.  (6 Marks)

### Q11. [Easy] When should one consider using Artificial Neural Networks for a problem? List any four situations, and state any two core ideas of the connectionism model.  (6 Marks)

## Module 3: Linear Neural Networks for Regression

### Q12. [Easy] Define regression and linear regression. Write the mathematical form of the linear regression model with d features and give two real-life examples of linear regression problems.  (6 Marks)

### Q13. [Easy] Which activation function is used when a single neuron implements linear regression? State its definition, any three properties, and explain why it is appropriate for regression.  (6 Marks)

### Q14. [Medium] Write the squared-error loss for a single example and the Mean Squared Error objective J(w) for a dataset of N examples (also in vector/matrix form). Give any three reasons why squared error is a good choice for linear regression.  (6 Marks)

### Q15. [Medium] Describe the batch gradient descent algorithm for linear regression. Give the gradient of the MSE loss in matrix form, the weight update rule, the effect of choosing the learning rate too large or too small, and two stopping criteria.  (6 Marks)

### Q16. [Tough] A tiny dataset for predicting house price from size is: (x=1, y=2), (x=2, y=4), (x=3, y=5). Using the model ŷ = w₀ + w₁x with initial weights w⁽⁰⁾ = [0, 0]ᵀ, learning rate η = 0.1 and loss J(w) = (1/2N)‖Xw − y‖², perform ONE iteration of batch gradient descent: compute the initial loss, the gradient, the updated weights w⁽¹⁾, and the new loss. Comment on the progress.  (6 Marks)

### Q17. [Medium] Explain the evaluation of a regression model. Define training vs test error, give the formulas for MSE, RMSE and MAE, and interpret the R² (coefficient of determination) score including the meaning of R² = 1, R² = 0 and R² < 0.  (6 Marks)

## Module 4: Linear Neural Networks for Classification

### Q18. [Easy] Define classification and distinguish it from regression. Explain the three types of classification with one example each.  (6 Marks)

### Q19. [Easy] Why is linear regression unsuitable for binary classification? Define the sigmoid function, sketch its shape, and state any four of its properties including its derivative.  (6 Marks)

### Q20. [Medium] Write the binary cross-entropy loss for a single example and the total loss over N examples. Explain how the loss behaves when the true label is 1 and when it is 0, and give any three advantages of cross-entropy over squared error for classification.  (6 Marks)

### Q21. [Medium] Compare batch gradient descent, stochastic gradient descent (SGD) and mini-batch SGD. Give two benefits and two challenges of SGD, and explain the terms batch size, iteration and epoch (with the formula for iterations per epoch).  (6 Marks)

### Q22. [Tough] A logistic regression model classifies pass/fail from study hours: dataset (x=1,y=0), (x=2,y=0), (x=3,y=1), (x=4,y=1). With w⁽⁰⁾ = [0,0]ᵀ (bias, weight), η = 0.5 and gradient ∇ℓ = (ŷ − y)x (x augmented with 1), perform two SGD iterations: first on example (x=1, y=0), then on example (x=3, y=1). Show forward pass, error, gradient and updated weights at each step.  (6 Marks)

### Q23. [Medium] Define the softmax function and list any three of its properties. Given logits z = [2.0, 1.0, 0.1] for a 3-class problem, compute the softmax probabilities and the predicted class.  (6 Marks)

### Q24. [Medium] For multi-class classification with K classes and d features: (i) explain one-hot encoding and give two reasons it is preferred; (ii) give the size of the weight matrix W and total parameter count; (iii) write the categorical cross-entropy loss and show its connection to binary cross-entropy when K = 2.  (6 Marks)

### Q25. [Tough] A binary classifier tested on 8 examples produced TP = 3, TN = 3, FP = 1, FN = 1. (i) Draw the confusion matrix and compute Accuracy, Precision, Recall and F1 score. (ii) State, with a practical example each, when Precision and when Recall is the more important metric, and why accuracy can be misleading.  (6 Marks)

## Module 5: Deep Feed-Forward Neural Networks and Backpropagation

### Q26. [Easy] What is non-linearity in a neural network and why is it essential? What happens to a deep network if all activations are linear (identity)? Define a non-linear function formally.  (6 Marks)

### Q27. [Easy] Compare the sigmoid, tanh and ReLU activation functions with respect to range, zero-centredness, vanishing-gradient behaviour and computational cost. Why is ReLU the default for hidden layers, and what is the “dying ReLU” problem?  (6 Marks)

### Q28. [Medium] Explain how a multilayer perceptron (2 inputs – 2 hidden units – 1 output) solves the XOR problem when a single neuron cannot. Refer to the roles of the hidden layer and the non-linear activation.  (6 Marks)

### Q29. [Medium] For a DFNN with L layers, write the layer-wise forward computation (pre-activation and activation) and the parameter-count formula for layer ℓ. Hence compute the total number of parameters of the network 784 → 256 → 128 → 10, showing the calculation per layer.  (6 Marks)

### Q30. [Tough] Derive/state the backpropagation equations for a DFNN trained with mini-batch SGD: (i) the output-layer error δ⁽ᴸ⁾ for the three standard loss–activation pairs (MSE+identity, BCE+sigmoid, CCE+softmax); (ii) the recursive backpropagation of the error signal to hidden layers; (iii) the weight and bias gradients for layer ℓ.  (6 Marks)

### Q31. [Tough] For a two-layer feedforward network of sigmoid units trained with squared-error loss L = ½Σₖ(tₖ − aₖ)², derive the error terms used in the backpropagation weight-update rule Δw = η·δ·x: (i) δₖ = aₖ(1−aₖ)(tₖ−aₖ) for an output unit k; (ii) δₕ = aₕ(1−aₕ)Σₖ δₖ wₖₕ for a hidden unit h.  (6 Marks)

### Q32. [Medium] Explain why backpropagation is computationally efficient compared to the naive approach of computing each parameter’s gradient independently. Quantify the speed-up for a network with P = 200,000 parameters.  (6 Marks)

### Q33. [Medium] Discuss the three main problems that arise when training very deep networks – vanishing gradients, exploding gradients and the degradation problem – with one solution for each. Also state the Universal Approximation Theorem and explain why deep networks are still preferred over wide shallow ones.  (6 Marks)

### Q34. [Easy] Design the output layer of a DFNN for (i) regression, (ii) binary classification and (iii) multi-class classification with C classes – giving the number of units, activation and loss for each. Also give the standard diagnosis and one remedy for: (a) loss = NaN/Inf, (b) training loss decreasing but validation loss increasing.  (6 Marks)

## Module 6: Convolutional Neural Networks

### Q35. [Easy] Why are fully connected networks poorly suited to images, and how do CNNs address this? State the three key principles of CNNs and briefly explain each.  (6 Marks)

### Q36. [Easy] Define the 2-D convolution (cross-correlation) operation as used in deep learning, and define feature map, kernel/filter and receptive field. What do high and low values in a feature map indicate?  (6 Marks)

### Q37. [Medium] Write the output-size formula for a convolutional layer with input size n, kernel k, padding p and stride s. Compute the output size of: (i) a 6×6 input, 3×3 kernel, s = 1, p = 0; (ii) the same with s = 3; (iii) state the “same” padding needed for a k×k kernel (s = 1) and explain why padding is used.  (6 Marks)

### Q38. [Medium] Give the formula for the number of parameters of a convolutional layer (kernel K×K, C_in input channels, C_out filters, with bias). Compute the parameters for: (i) first layer with 224×224×3 input, 5×5 kernels, 64 filters; (ii) a deeper layer with 56×56×128 input, 3×3 kernels, 256 filters. How many learnable parameters does a 2×2 max-pooling layer have, and why?  (6 Marks)

### Q39. [Tough] Define receptive field and give the formula RF_l = RF_{l−1} + (K_l − 1)·Π S_i. For a stack of four 3×3 convolutions with stride 1, compute the receptive field after each layer. Hence argue why stacking small 3×3 kernels is preferred over one large 9×9 kernel (consider both receptive field and parameters).  (6 Marks)

### Q40. [Medium] Given the 4×4 input [[2,4,7,4],[6,6,9,8],[3,4,8,3],[7,5,3,6]], compute the output of (i) 2×2 max pooling with stride 2 and (ii) 2×2 average pooling with stride 2. State any three purposes of pooling layers.  (6 Marks)

### Q41. [Medium] What does a 1×1 convolution compute? Give two of its key applications. In a bottleneck design, replacing a direct 5×5 convolution on 256 channels (256 output channels) with 1×1 (256→64) → 5×5 (64→64) → 1×1 (64→256) reduces parameters from 1,638,400 to 147,456 – verify these counts (ignore biases) and state the reduction factor.  (6 Marks)

### Q42. [Tough] Compare strided convolution and dilated (atrous) convolution with respect to output size, receptive field, parameters and resolution loss, and give one typical use case for each. For a 3×3 kernel with dilation rate r = 2, compute the effective receptive field using k_eff = k + (k−1)(r−1).  (6 Marks)

### Q43. [Medium] Describe the typical CNN architecture pattern INPUT → [CONV → ACTIVATION → POOL]×N → FC → OUTPUT, explaining what early, middle and deep layers learn (hierarchical feature learning). What are skip/residual connections and give two benefits.  (6 Marks)

## Module 7: Recurrent Neural Networks

### Q44. [Easy] Why do feedforward networks fail on sequential data? Define a Recurrent Neural Network and state the three main advantages of parameter sharing across time steps.  (6 Marks)

### Q45. [Medium] Write the mathematical formulation of a vanilla RNN (hidden-state update and output equations) defining every symbol. Derive the parameter-count formula and evaluate it for input dimension d = 100, hidden dimension h = 256 and vocabulary V = 10,000.  (6 Marks)

### Q46. [Tough] A vanilla RNN has d = 2 inputs, h = 3 hidden units, V = 2 outputs, φ = tanh, h₀ = [0,0,0]ᵀ, with W_hx = [[0.5,0.3],[0.2,0.4],[0.1,0.2]], W_hh = [[0.3,0.2,0.1],[0.1,0.4,0.2],[0.2,0.1,0.3]], b_h = [0.1,0.1,0.1]ᵀ, W_ho = [[0.4,0.3,0.5],[0.2,0.6,0.1]], b_o = [0.1,0.1]ᵀ. For input x₁ = [1.0, 0.5]ᵀ, compute z₁, h₁ and o₁ (show all steps).  (6 Marks)

### Q47. [Medium] Explain Backpropagation Through Time (BPTT). Include: the idea of unrolling, the total sequence loss, the recursive hidden-state gradient ∂L/∂h_t, and why parameter gradients are sums over time steps.  (6 Marks)

### Q48. [Tough] Explain the vanishing-gradient problem in vanilla RNNs. Show that ∂L/∂h₁ involves the product ∏ diag(φ′(z_t))·W_hh over time. With ‖W_hh‖ ≈ 0.8 and |φ′| ≈ 0.5, compute the gradient attenuation after 5, 10 and 20 steps, and state two consequences and two solutions.  (6 Marks)

### Q49. [Medium] Describe the GRU cell: write the equations of the reset gate, update gate, candidate hidden state and final hidden state, explaining the role of each. Why does the GRU mitigate vanishing gradients, and what is its parameter count formula (excluding output layer)?  (6 Marks)

### Q50. [Medium] Describe the LSTM cell: name its three gates and two states, write the cell-state and hidden-state update equations, and explain why the LSTM solves the vanishing-gradient problem. Compare LSTM with GRU on any four points and state when each is preferred.  (6 Marks)


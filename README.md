# iNaturalist_12k
## **TASK-1** <br />
### Training & Testing
I would proceed with the keras multi-class classification method on the dataset provided. <br />
Training the model includes the following: <br />
**Data preparation :** Splitting the dataset into training and validation sets. Additionally, making sure the annotations are done properly and the input data is appropriately formatted. Standardization could be used on the input data for normalization and thus, achieving faster runs. <br />
**Defining the model :** The neural network model will be built with Keras. The output layer of the model will have multiple neurons for multi-class classification where each neuron will represent one class. Softmax could be used as an activation function to produce class probabilities. <br />
**Label Encoding :** A multi-class classification would require label encoding for more efficient working. Label encoding is necessary because most machine learning algorithms operate on numerical data and mathematical operations are performed on these numbers to make predictions and learn patterns. <br />
**Compile the model :** Specify the loss function, optimizer and evaluation metric for configuring the learning process of the model. Categorical cross-entropy loss function is used majorly for multi-class classification. <br />
**Defining network hyperparameters :** Network hyperparameters can significantly impact the model's performance and training process. Parameters including number of convolutional layers, kernel size, learning rate, number of epochs, early stop, etc. are initialized and often require tuning based on the results achieved while testing on the validation datset. <br />
**Training the model :** The model is trained on the training dataset using the 'fit' method. The model learns to map the input data to the corresponding class while training. <br />
**Evaluating the model :** Finally, the model is evaluated for accuracy on the basis of its performance on the validation sets. <br />

**Model Testing :** The evaluation of the model could be done using various testing methods, namely : Confusion matrix, ROC curve, F1 score, AUC value amongst others. <br />










  						  

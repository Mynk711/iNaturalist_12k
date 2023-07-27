# iNaturalist_12k
## **TASK-1** <br />
### Training & Testing
I would proceed with the keras multi-class classification method on the dataset provided. <br />
Training the model includes the following: <br /> <br />
**Data preparation :** Splitting the dataset into training, validation and test sets. Additionally, making sure the annotations are done properly and the input data is appropriately formatted. Standardization could be used on the input data for normalization and thus, achieving faster runs. <br /><br />
**Defining the model :** The neural network model will be built with Keras. The output layer of the model will have multiple neurons for multi-class classification where each neuron will represent one class. Softmax could be used as an activation function to produce class probabilities. <br /><br />
**Label Encoding :** A multi-class classification would require label encoding for more efficient working. Label encoding is necessary because most machine learning algorithms operate on numerical data and mathematical operations are performed on these numbers to make predictions and learn patterns. <br /><br />
**Compile the model :** Specify the loss function, optimizer and evaluation metric for configuring the learning process of the model. Categorical cross-entropy loss function is used majorly for multi-class classification. <br /><br />
**Defining network hyperparameters :** Network hyperparameters can significantly impact the model's performance and training process. Parameters including number of convolutional layers, kernel size, learning rate, number of epochs, early stop, etc. are initialized and often require tuning based on the results achieved while testing on the validation datset. <br /><br />
**Training the model :** The model is trained on the training dataset using the 'fit' method. The model learns to map the input data to the corresponding class while training. <br /><br />
**Evaluating the model :** Finally, the model is evaluated for accuracy on the basis of its performance on the test sets. <br />

**Model Testing :** The evaluation of the model could be done using various testing methods, namely : Confusion matrix, ROC curve, F1 score, AUC value amongst others. <br /><br />

## **Task-2** <br />
After building the initial model, we need to identify the hyperparameters that needs tuning and find the best combination for improved efficiency and performance.<br />
In order to evaluate the impact of hyperparameters on the model's accuracy, we need to log them. <br /><br />
**Logging the hyperparameters-** We can either use custom logging where the data is saved in a json or a text file and can be used in later iterations of the model.
TenserBoard could also be used with Keras for logging these hyperparameters. <br /><br />
**Evaluation of Hyperparameters :** Consider different values and combinations of hyperparameters (learning rate, number of epochs, batch sizes, etc.) that needs evaluation. <br />
We train and evaluate the model with different hyperparameter combinations on the validation set. <br />
We select an evaluation metric which aligns with the problem statement. It could be accuracy, recall, AUC value, etc. In this scenario, the prime objective is to identify the class of the image given as input, so we will focus on the accuracy of the model. <br />
We'll keep track of the evaluation metric values for each hyperparameter combination and record model's accuracy on the validation dataset for all experiments.<br />
The hyperparameter combination that leads to the best performance on the validation set will be used for the final model.<br />
The model is trained using the hyperparameters established previously on the combined traing and validation sets.<br />
Finally, the performance of the model is judged on the basis of its result on the test dataset, which gives an unbiased estimate of how well the model can generalize to unseen, new data.<br />
We need to iterate the hyperparameter tuning process and fine tune it to achieve more accuracy and efficiency.<br />












  						  

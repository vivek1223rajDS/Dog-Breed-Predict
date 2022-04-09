# Dog Breed Prediction Project

This is a Udacity Data Scientist Nanodegree Capstone project where we have built a dog breed classifier app in ipython notebook. We can pass an image of dog to the app and it will return the breed of the dog. The Deep Learning model distinguishes between the 133 classes of dogs with an accuracy of over 80.3828%.

Medium link - https://medium.com/@raj.vivek1223/dog-breed-classification-c35ed6856a46

## Project details:
The task is to develop an algorithm that takes an image as an input, pre-processes, and transforms the image so that it can be fed into a CNN for classifying the breed of the dog. If a human image is uploaded, it should still tell the user what dog breed the human resembles most.

## Dataset details:

The datasets are provided by Udacity i.e. dog images for training the models and human faces for the detector. After loading both the dataset using sklearn, the following conclusions are drawn:

1. There are 133 total dog categories.
2. There are 8351 total dog images.
3. There are 6680 training dog images.
4. There are 835 validation dog images.
5. There are 836 test dog images.
6. There are in total 13233 human images.

## Metrics

This model was evaluated using metric known as accuracy. Why we choose accuracy as a mertic to evaluate is based on the fact of no class imbalance issue in the data. I have broken down the training set according to the breeds of the dog. It appears that there is no problem of class imbalance. Therefore, accuracy is utilized as the metric to evaluate the CNN model which will be trained in this exercise.

## Libraries Used:

1. OpenCV
2. h5py
3. Matplotlib
4. Numpy
5. Scipy
6. tqdm
7. Keras
8. Scikit-learn
9. Pillow
10. ipykernel
11. TensorFlow

## Description of repository:
The repository consists of the Jupyter Notebook files from the Udacity classroom, in both formats: Dog_Breed_Predict_App.html and Dog_Breed_Predict_App.ipynb. All credits for code examples here go to Udacity.

## Steps Involved:

1. Import Datasets
2. Detect Humans
3. Detect Dogs
4. Create a CNN to Classify Dog Breeds (from Scratch)
5. Use a CNN to Classify Dog Breeds (using Transfer Learning)
6. Create a CNN to Classify Dog Breeds (using Transfer Learning)
7. Write your Algorithm
8. Test Your Algorithm

## Analysis of the Project:

I decided to use a pre-trained ResNet50 model as this has shown very good results with regard to accuracy for image classification. In the provided classroom environment, my tests showed an a test accuracy of 80.3828%. This was accomplished by 25 epochs which ran very quickly on the provided GPU. Thanks to Udacity! The code in the classroom worked pretty well.


### Review:

1. An example of human detection is provided in the following image:

Human is detected in the following image.

![44](https://user-images.githubusercontent.com/34116562/82108644-89e53f80-974d-11ea-9661-2dd62a57e023.png)


2. Even humans will find it difficult to tell the difference between the two dog classes in some categories. An example is shown below:

![Brittany_02625](https://user-images.githubusercontent.com/34116562/82108456-1db60c00-974c-11ea-89c9-c4397c8bc57b.jpg)

Brittany Breed

![Welsh_springer_spaniel_08203](https://user-images.githubusercontent.com/34116562/82108457-1f7fcf80-974c-11ea-9d4f-6ec00b36b05c.jpg)

Welsh Springer Spaniel Breed

3. Also, more distinguishing/challenging categories are shown.

![final](https://user-images.githubusercontent.com/34116562/82108643-88b41280-974d-11ea-86f9-f64ee078518a.png)


## Conclusion:

I started with a relatively small dataset containing different images of dog breeds and built a CNN from scratch that had 3 convolutional layers with pooling, the accuracy on the test set was 9.0909%. Then I used pre-trained convolutional networks provided by Keras. On the test set, VGG16 gave 43.1818% and ResNet50 gave 80.3828% accuracy. The results are satisfying but I believe they can be improved further using more pre-trained models and maybe tweaking hyperparameters of currently used models.
Before starting the project, my objective was to create a CNN with 90% testing accuracy. My final model testing accuracy was around 80%.

A few possible points of improvement for our algorithm could be;

  1. We can try to change the architecture of layers, or use more fully connected layers and a deeper network, although it might not necessarily improve the results.
  2. We can use the GridSearch function to tune the hyperparameters further and get better results. Parameters such as optimizer, loss function, activation function, epoch, etc.
  3. We can try different dropout layers and dropout rates in order to reduce overfitting and achieve more accurate testing results.
  4. We can improve the training process by improving our data set by augmentation (so our model can be robust to image scaling, translation, occlusion, etc.)

Following the above areas, I’m confident enough that we can increase the testing accuracy of the model to above 90%.

## Results:

Using the final model, some examples of predictions are shown below. If a photo of a human is uploaded, it tells the closest match.

#### Prediction: This photo looks like an Afghan hound.

![1](https://user-images.githubusercontent.com/34116562/82108536-bc426d00-974c-11ea-9c9e-eea43de57701.png)

#### Prediction: The predicted dog breed is a Brittany.

![2](https://user-images.githubusercontent.com/34116562/82108537-be0c3080-974c-11ea-9d92-f73a314f70f0.png)

#### Prediction: The predicted dog breed is a Boykin spaniel.

![3](https://user-images.githubusercontent.com/34116562/82108538-bfd5f400-974c-11ea-9426-3437ace3342a.png)

#### Prediction: The predicted dog breed is a Curly-coated retriever.

![4](https://user-images.githubusercontent.com/34116562/82108540-c19fb780-974c-11ea-9a01-6ad7f33d98cc.png)

#### Prediction: The predicted dog breed is a Labrador retriever.

![5](https://user-images.githubusercontent.com/34116562/82108545-c5333e80-974c-11ea-9b21-8876e669061b.png)

#### Prediction: The predicted dog breed is a Labrador retriever.

![6](https://user-images.githubusercontent.com/34116562/82108549-c82e2f00-974c-11ea-98dc-4372bde8627d.png)

#### Prediction: The predicted dog breed is a Labrador retriever.

![7](https://user-images.githubusercontent.com/34116562/82108551-ca908900-974c-11ea-938f-8dfd4bb95c17.png)

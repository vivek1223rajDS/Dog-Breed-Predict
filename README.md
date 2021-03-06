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

An example of human detection is provided in the following image:

Human is detected in the following image.

![CNN1](https://user-images.githubusercontent.com/77801625/162848490-6ea01367-629b-4db6-95c6-79ca8d59c87e.png)

## Conclusion:

We started with building a CNN from scratch that had 3 convolutional layers with pooling, the accuracy on the test set was 9.0909%. Following that we used pre-trained VGG16 which gave 43.1818%, then finally we build the best model ResNet50 which gave 80.3828% accuracy. The data was relatively small containing different images of dog breeds but the results are satisfying. I believe they can be improved further using more pre-trained models and maybe tweaking hyperparameters of currently used models.

A few possible points of improvement for our algorithm could be;
  * Try changing the architecture of layers, or use more fully connected layers and a deeper network, although it might not necessarily improve the results.
  * We can try different dropout layers and dropout rates in order to reduce overfitting and achieve more accurate testing results. We tried some of them and it improved the accuracy as well.
  * We can use the GridSearch function to tune the hyperparameters further and get better results. Parameters such as optimizer, loss function, activation function, epoch, etc.
  * We can improve the training process by improving our data set by augmentation (so our model can be robust to image scaling, translation, occlusion, etc.)


## Results:

Using the final model, some examples of predictions are shown below. If a photo of a human is uploaded, it tells the closest match.

#### Prediction: This photo looks like an Afghan hound.

![CNN2](https://user-images.githubusercontent.com/77801625/162848555-ba8f442c-d18c-432d-9b3f-57598f1a55bf.png)

#### Prediction: The predicted dog breed is a Bichon frise.

![CNN3](https://user-images.githubusercontent.com/77801625/162848707-2660129c-9ac8-48be-9131-5aab22889354.png)

#### Prediction: The predicted dog breed is an Irish water spaniel.

![CNN4](https://user-images.githubusercontent.com/77801625/162848781-5d0438c7-e7c3-45b8-a2a6-26fa4997959a.png)

#### Prediction: The predicted dog breed is a Flat-coated retriever.

![CNN5](https://user-images.githubusercontent.com/77801625/162848870-f33df064-1e63-4e95-8026-70de9cd38a25.png)


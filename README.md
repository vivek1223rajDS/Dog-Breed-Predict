# Dog Breed Classification Project

This is a Udacity Data Scientist Nanodegree Capstone project where we have built a dog breed classifier app in ipython notebook. We can pass an image of dog to the app and it will return the breed of the dog. The Deep Learning model distinguishes between the 133 classes of dogs with an accuracy of over 80.3828%.

## Project details:
The task was to develop an algorithm that takes an image as an input, pre-processes and transforms the image so that it can be fed into a CNN for classifying the breed of the dog. If a human image is uploaded, it should still tell the user what dog breed the human resembles most.

### Dataset details:

The datasets are provided by Udacity i.e. dog images for training the models and human faces for detector. After loading both the dataset using sklearn, the following conclusions are drawn:

1. There are 133 total dog categories.
2. There are 8351 total dog images.
3. There are 6680 training dog images.
4. There are 835 validation dog images.
5. There are 836 test dog images.
6. The are in total 13233 human images.

## Steps Involved:

1. Import Datasets
2. Detect Humans
3. Detect Dogs
4. Create a CNN to Classify Dog Breeds (from Scratch)
5. Use a CNN to Classify Dog Breeds (using Transfer Learning)
6. Create a CNN to Classify Dog Breeds (using Transfer Learning)
7. Write your Algorithm
8. Test Your Algorithm

## Libraries Used:

1. opencv-python
2. h5py
3. matplotlib
4. numpy
5. scipy
6. tqdm
7. keras
8. scikit-learn
9. pillow
10. ipykernel
11. tensorflow


## Description of repository:
The repository consists of the Jupyter Notebook files from the Udacity classroom, in both formats: Dog_Breed_Predict_App.html and Dog_Breed_Predict_App.ipynb. All credits for code examples here go to Udacity.


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
I was surprised by the good results of the algorithm i.e. Resnet50. Without doing too much fine-tuning, the algorithm was already providing high accuracy and the predictions were mostly correct. An accuracy of over 80%. For human faces it seems easier if the face has distinct features that resembles a certain dog breed. Otherwise, it starts to guess from some features, but the results vary. For higher accuracy, the parameters could be further optimized, maybe also including more layers into the model. Further, number of epochs could be increased to 40 to lower the loss. Also by providing an even bigger training data set, the classification accuracy could be improved further. Another improvement could be made with regard to UI.

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

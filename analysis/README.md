#Classification of MNIST-Data

[Kaggle-Competition](https://www.kaggle.com/c/digit-recognizer)
Current Highscore: 98.8%

## Implementing various ML/DL algorithms - Performance Assessment
### logistic Regression:
* achieved Train Accuracy of 91% with reg = .3
* accuracy largely equal for reg .1 - 20

### SVM
* achieved Train Accuracy of 91% with c2 = .1 - 4
* best c2 = .1

### KNN
* train accuracy with n=11 is 0.96
* BEST RESULT SO FAR!

### Random Forest:
* Number of Trees K: K = 30 -> 95%, K = 100 -> 96%, K = 1000 - >96.5% ON TRAINING SET! 

### Fully connected Net, 3 Layers - 100 Neurons, relu, relu, softmax:
* after 100 epochs, train accuracy was 99.8%, but test accuracy was 97%

## Bagging ML Algs on Test Data - only minor improvement
* svm, knn, svm, rf -> 96.6%
* svm, knn, svm, rf, nn -> 97.4%


## Learnings:
* a neural net performs better when PCA + Mean Subtraction is applied
* Was able to achieve a dim. reduction to 30px and still get 99.3% train acc, 95% test acc, 15px - > 97%, 7px -> 88.8% train accuracy.
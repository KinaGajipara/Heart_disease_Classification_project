            
                                       # classification Project

library(caret)
library(rsample)
library(modeldata)
library(mclust)
library(tidyverse)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(keras)
library(neuralnet)
library(factoextra)
library(glmnet)
library(corrplot)
library(MLmetrics)
library(pROC)

heart_attack_train <- read.csv("C:/Users/divya/Desktop/conestoga/multivariate/Heart disease 1.csv")
heart_attack_test <-  read.csv("C:/Users/divya/Desktop/conestoga/multivariate/Heart disease 2.csv")

#Data Preprocessing

# checking missing counts in both train and test datasets.
missing_count <- sum(is.na(heart_attack_train))  # Count missing values in each column
View(missing_count)
missing_count <- sum(is.na(heart_attack_test))  # Count missing values in each column
View(missing_count)

# removing X, which shows index in test dataset. 
heart_attack_test <- heart_attack_test[, -which(names(heart_attack_test) == "X")]

# create summary of train dataset
summary(heart_attack_train)

# ploting correlation table
numeric_data <- select_if(heart_attack_train, is.numeric)
correlation_matrix <- cor(numeric_data)
print(correlation_matrix)
corrplot(correlation_matrix, method = "color", tl.cex = 0.5, addCoef.col = "black") 

# factorizing target variable
heart_attack_train$target <- factor(heart_attack_train$target, levels = c(0, 1))
heart_attack_test$target <- factor(heart_attack_test$target, levels = c(0, 1))


# creating Logistic model
logistic_heart_disease_model <- train(
  target ~ . ,                                   
  data = heart_attack_train,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)
logistic_heart_disease_model


# Prediction on Logistic model
predict_logit <- predict(logistic_heart_disease_model, newdata = heart_attack_test, type = "prob")
predicted_classes <- ifelse(predict_logit[,"1"] > 0.5, "1", "0")

predicted_classes <- factor(predicted_classes, levels = levels(heart_attack_test$target))
logistic_conf_matrix <- confusionMatrix(predicted_classes, heart_attack_test$target)
logistic_conf_matrix
logistic_accuracy <- logistic_conf_matrix$overall["Accuracy"]
logistic_accuracy
logistic_precision <- precision(predicted_classes, heart_attack_test$target)
logistic_precision
logistic_recall <- recall(predicted_classes, heart_attack_test$target)
logistic_recall
logistic_F1score <- F1_Score(predicted_classes, heart_attack_test$target)
logistic_F1score

conf_matrix_logistic <- as.data.frame(logistic_conf_matrix$table)

ggplot(data = conf_matrix_logistic, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "black") +
  geom_text(aes(label = Freq)) +
  theme_minimal() +
  scale_fill_gradient(low = "lightgreen", high = "steelblue") +
  labs(title = "Confusion Matrix",
       x = "Actual",
       y = "Predicted")


# creating KNN model

knn_hyperparameters <- expand.grid(k = c(3, 5, 7, 9, 11))

knn_model <- train(target ~ ., 
                   data = heart_attack_train,
                   method = "knn", 
                   preProcess = c("center","scale"),
                   trControl = trainControl(method = "cv", number = 10),
                   tuneGrid = knn_hyperparameters)
knn_model


#prediction on KNN model
knn_prediction <- predict(knn_model, newdata = heart_attack_test)

knn_conf_matrix <- confusionMatrix(knn_prediction, heart_attack_test$target)
knn_conf_matrix
knn_accuracy <- knn_conf_matrix$overall["Accuracy"]
knn_accuracy
knn_precision <- precision(knn_prediction, heart_attack_test$target)
knn_precision
knn_recall <- recall(knn_prediction, heart_attack_test$target)
knn_recall
knn_F1score <- F1_Score(knn_prediction, heart_attack_test$target)
knn_F1score

conf_matrix_knn <- as.data.frame(knn_conf_matrix$table)

ggplot(data = conf_matrix_knn, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "black") +
  geom_text(aes(label = Freq)) +
  theme_minimal() +
  scale_fill_gradient(low = "lightgreen", high = "steelblue") +
  labs(title = "Confusion Matrix",
       x = "Actual",
       y = "Predicted")


# creating decision tree

decision_tree_model <- train(target ~ .,
                       data = heart_attack_train,
                       method = "rpart",
                       tuneGrid = expand.grid(cp = seq(0.01, 0.5, by = 0.01)),
                       control = rpart.control(maxdepth = 5)
)

rpart.plot(decision_tree_model$finalModel)
decision_tree_model

#Prediction on Decision tree
decision_tree_predictions <- predict(decision_tree_model, newdata = heart_attack_test)

decision_tree_conf <- confusionMatrix(decision_tree_predictions, heart_attack_test$target)
decision_tree_conf
dicision_tree_accuracy <- decision_tree_conf$overall["Accuracy"]
dicision_tree_accuracy
dicision_tree_precision <- precision(decision_tree_predictions, heart_attack_test$target)
dicision_tree_precision
decision_tree_recall <- recall(decision_tree_predictions, heart_attack_test$target)
decision_tree_recall
decision_tree_F1score <- F1_Score(decision_tree_predictions, heart_attack_test$target)
decision_tree_F1score

conf_matrix_decision_tree <- as.data.frame(decision_tree_conf$table)

ggplot(data = conf_matrix_decision_tree, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "black") +
  geom_text(aes(label = Freq)) +
  theme_minimal() +
  scale_fill_gradient(low = "lightgreen", high = "steelblue") +
  labs(title = "Confusion Matrix",
       x = "Actual",
       y = "Predicted")


# creating random forest model

levels(heart_attack_train$target)
heart_attack_train$target <- factor(heart_attack_train$target, labels = c("no", "yes"))
heart_attack_test$target <- factor(heart_attack_test$target, labels = c("no", "yes"))

random_forest_model <- train(target ~ .,
                              data = heart_attack_train,
                              method = "ranger",
                              trControl = trainControl(method="cv",
                                                       number = 5,
                                                       verboseIter = TRUE,
                                                       classProbs = TRUE),
                              num.trees = 100,
                              importance = "impurity"
                             )
random_forest_model
varImp(random_forest_model)

#Prediction on random forest model

random_forest_predicion <- predict(random_forest_model, newdata = heart_attack_test)

random_forest_Conf <- confusionMatrix(random_forest_predicion , heart_attack_test$target)
random_forest_Conf
random_forest_accuracy <- random_forest_Conf$overall["Accuracy"]
random_forest_accuracy
random_forest_precision <- precision(random_forest_predicion, heart_attack_test$target)
random_forest_precision
random_forest_recall <- recall(random_forest_predicion, heart_attack_test$target)
random_forest_recall
random_forest_F1score <- F1_Score(random_forest_predicion, heart_attack_test$target)
random_forest_F1score

conf_matrix_random_forest <- as.data.frame(random_forest_Conf$table)

ggplot(data = conf_matrix_random_forest, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "black") +
  geom_text(aes(label = Freq)) +
  theme_minimal() +
  scale_fill_gradient(low = "lightgreen", high = "steelblue") +
  labs(title = "Confusion Matrix of random forest",
       x = "Actual",
       y = "Predicted")


#creating neural network

model <- neuralnet(target ~ .,
                  data=heart_attack_train,
                  hidden=c(4,2),
                  linear.output = FALSE
)
plot(model,rep = "best")


#prediction on neural network

neural_predict <- predict(model, heart_attack_test)
neural_predict
neural_predict_prob <- compute(model, heart_attack_test)$net.result
neural_prediction <- ifelse(neural_predict[,1] > 0.5, "yes", "no")
neural_prediction <- factor(neural_prediction, levels = c("no", "yes"))

neural_Conf <- confusionMatrix(neural_prediction , heart_attack_test$target)
neural_Conf
neural_accuracy <- neural_Conf$overall["Accuracy"]
neural_accuracy
neural_precision <- neural_Conf$byClass["Precision"]
neural_precision
neural_recall <- neural_Conf$byClass["Recall"]
neural_recall
neural_F1score <- neural_Conf$byClass["F1"]
neural_F1score

conf_matrix_neural <- as.data.frame(neural_Conf$table)

ggplot(data = conf_matrix_neural, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "black") +
  geom_text(aes(label = Freq)) +
  theme_minimal() +
  scale_fill_gradient(low = "lightgreen", high = "steelblue") +
  labs(title = "Confusion Matrix",
       x = "Actual",
       y = "Predicted")


#Calculating AUC and plotting ROC curves

logistic_roc <- roc(heart_attack_test$target, predict_logit[,"1"])
auc_value_logistic <- auc(logistic_roc)
auc_value_logistic

knn_probabilities <- predict(knn_model, newdata = heart_attack_test, type = "prob")
knn_prob_yes <- knn_probabilities[, "1"]
knn_roc <- roc(heart_attack_test$target, knn_prob_yes)
auc_value_KNN <- auc(knn_roc)
auc_value_KNN

decision_probabilities <- predict(decision_tree_model, newdata = heart_attack_test, type = "prob")
decision_prob_yes <- decision_probabilities[, "1"]
tree_roc <- roc(heart_attack_test$target, decision_prob_yes)
auc_value_tree <- auc(tree_roc)
auc_value_tree

random_forest_probabilities <- predict(random_forest_model, newdata = heart_attack_test, type = "prob")
random_forest_prob_yes <- random_forest_probabilities[, 1]
forest_roc <- roc(heart_attack_test$target, random_forest_prob_yes)
auc_value_forest <- auc(forest_roc)
auc_value_forest

neural_predict_prob <- neural_predict_prob[, 2]
neural_roc <- roc(heart_attack_test$target, neural_predict_prob)
auc_value_neural <- auc(neural_roc)
auc_value_neural

# Plot ROC curves

plot(forest_roc, col = "blue", main = "ROC Curves")
lines(knn_roc, col = "red")
lines(tree_roc, col = "black")
lines(logistic_roc, col = "orange")
lines(neural_roc, col = "purple")

# Add legend

legend("bottomright", legend = c("Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Neural Network"), 
       col = c("blue", "red", "black", "orange", "purple"), lty = 1)














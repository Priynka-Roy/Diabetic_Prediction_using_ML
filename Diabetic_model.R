install.packages(c("tidyverse", "caret", "randomForest", "e1071", "ggplot2", "pROC" ))
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(ggplot2)
library(pROC)


# Load dataset
df <- read.csv("C:\\Users\\priyn\\OneDrive\\Desktop\\Gproject\\1 diabetes.csv")

# View first few rows
head(df)

# Check dataset structure
str(df)

# Summary statistics
summary(df)

# Check for missing values
sum(is.na(df))

# Convert Outcome to factor (1 = Diabetic, 0 = Non-Diabetic)
df$Outcome <- as.factor(df$Outcome)

# Normalize numerical variables
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

df[ ,1:8] <- as.data.frame(lapply(df[ ,1:8], normalize))

# Split data into training (80%) and testing (20%) sets
set.seed(123)
index <- createDataPartition(df$Outcome, p = 0.8, list = FALSE)
train_data <- df[index, ]
test_data <- df[-index, ]

# Check dimensions
dim(train_data)
dim(test_data)

# Train logistic regression model
log_model <- glm(Outcome ~ ., data = train_data, family = binomial)

# Model summary
summary(log_model)

# Predict on test data
log_pred <- predict(log_model, test_data, type = "response")
log_pred <- ifelse(log_pred > 0.5, 1, 0)

# Convert predictions to factor
log_pred <- as.factor(log_pred)

# Evaluate model
confusionMatrix(log_pred, test_data$Outcome)

# Train Random Forest model
rf_model <- randomForest(Outcome ~ ., data = train_data, ntree = 100)

# Predict on test data
rf_pred <- predict(rf_model, test_data)

# Evaluate model
confusionMatrix(rf_pred, test_data$Outcome)

# Logistic Regression Performance
log_cm <- confusionMatrix(log_pred, test_data$Outcome)
log_cm$overall["Accuracy"]

# Random Forest Performance
rf_cm <- confusionMatrix(rf_pred, test_data$Outcome)
rf_cm$overall["Accuracy"]

varImpPlot(rf_model)

ggplot(df, aes(x = Outcome, fill = Outcome)) +
  geom_bar() +
  ggtitle("Diabetes Class Distribution") +
  theme_minimal()

saveRDS(rf_model, "diabetes_rf_model.rds")
loaded_model <- readRDS("diabetes_rf_model.rds")

# Compute ROC for Logistic Regression
log_roc <- roc(test_data$Outcome, as.numeric(log_pred))
rf_roc <- roc(test_data$Outcome, as.numeric(rf_pred))

# Plot ROC Curves
plot(log_roc, col="blue", main="ROC Curves for Models", lwd=2)
lines(rf_roc, col="red", lwd=2)
legend("bottomright", legend=c("Logistic Regression", "Random Forest"),
       col=c("blue", "red"), lwd=2)

# Get AUC values
auc(log_roc)
auc(rf_roc)

# Compare Accuracy Visually
accuracy_df <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(log_cm$overall["Accuracy"], rf_cm$overall["Accuracy"])
)

ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  ggtitle("Model Accuracy Comparison") +
  theme_minimal()












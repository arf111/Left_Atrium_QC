import pandas as pd

dict_of_model_with_loss_and_mse = {}

train_loss = [i for i in range(1, 26)]
val_loss = [i for i in range(1, 26)]
train_mse = [i for i in range(1, 26)]
val_mse = [i for i in range(1, 26)]

dict_of_model_with_loss_and_mse["resnet50_train_loss"] = train_loss
dict_of_model_with_loss_and_mse["resnet50_val_loss"] = val_loss
dict_of_model_with_loss_and_mse["resnet50_train_mse"] = train_mse
dict_of_model_with_loss_and_mse["resnet50_val_mse"] = val_mse
dict_of_model_with_loss_and_mse["resnet50_best_mse"] = 0.5

train_loss = [i for i in range(1, 26)]
val_loss = [i for i in range(1, 26)]
train_mse = [i for i in range(1, 26)]
val_mse = [i for i in range(1, 26)]

dict_of_model_with_loss_and_mse["resnet101_train_loss"] = train_loss
dict_of_model_with_loss_and_mse["resnet101_val_loss"] = val_loss
dict_of_model_with_loss_and_mse["resnet101_train_mse"] = train_mse
dict_of_model_with_loss_and_mse["resnet101_val_mse"] = val_mse
dict_of_model_with_loss_and_mse["resnet101_best_mse"] = 0.95

# make dataframe
df = pd.DataFrame(dict_of_model_with_loss_and_mse)
print(df.head())
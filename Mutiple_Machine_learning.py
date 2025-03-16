import torch.nn as nn
import torch
import pandas as pd  
import numpy as np  
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
torch.manual_seed(1)
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import seaborn as sns
from glob import glob
from torchvision import transforms
#from cv_methods import CVMethods
import cv2
EPOCH = 48             # train the training data n times
BATCH_SIZE = 128       # 一批训练的量
INPUT_SIZE = 4096        # input size 特征的数量
LR = 0.0001                # learning rate
count = 4              # 类别数量  
HIDDEN_SIZE = 64        # 隐藏层大小
SHUFFLE=True        # 是否打乱数据集
t0 = time()
class MyDataset(Dataset):#元组用的数据集，可以与常规的对比学习
        def __init__(self, data_list):
            self.data_list = data_list
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, index):
            item = self.data_list[index]
            features = torch.tensor(item[0], dtype=torch.float32)
            lable = torch.tensor(item[1], dtype=torch.int64)
            return features, lable
        
class CustomDataset(Dataset):#dataframe用的数据集
        def __init__(self, features, labels):
            self.features = torch.tensor(features.to_numpy(), dtype=torch.float32)
            self.labels = torch.tensor(labels.to_numpy(), dtype=torch.int64)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, index):
            return self.features[index], self.labels[index]

class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.rnn = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,  # 隐藏层大小
                num_layers=6,  # 增加LSTM层数
                batch_first=True,
                dropout=0.3    # 增加Dropout层防止过拟合
            )
            self.out = nn.Linear(HIDDEN_SIZE, count)
        def forward(self, x):
            r_out, (h_n, h_c) = self.rnn(x, None)
            out = self.out(r_out[:, -1, :])
            return out
        
# MLP模型定义
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, count)
    
    def forward(self, x): 
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
#自己定义的损失函数            
class CustomLoss(nn.Module):
        def __init__(self):
            super(CustomLoss, self).__init__()

        def forward(self, outputs, targets):
            # 将0号标签的预测置为0
            outputs[:, 0] = 0
            # 计算损失时忽略0号标签
            loss = F.cross_entropy(outputs, targets, ignore_index=0)
            return loss
            
class MutipleMethods:
  def __init__(self):
    pass

 
  def split_data_with_sklearn(self,df):#划分dataframe
    #分帧了之后，其实每个帧间隔就那么点，每个帧其实区别不大，
    #所以约等于训练集跟测试集一样一样的，导致了准确率虚高
    t0 = time()
    #shuffle=False 保证数据顺序不变 stratify=df['lable']保证训练集和测试集的标签分布相同
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1], 
                                                        df['lable'], 
                                                        test_size=0.3,random_state=42,
                                                        shuffle=SHUFFLE)
    tt = time() - t0
    print("Split dataset in {} seconds".format(round(tt, 3)))
    print(f"y_train value_counts is {y_train.value_counts()}, y_test value_counts is {y_test.value_counts()}")
    #print(f"y_train value_counts is {y_train['lable'].value_counts()}, y_test value_counts is {y_test['lable'].value_counts()}")
    return X_train,X_test,y_train,y_test 
  

  def split_data_with_size(self,path):#按文件大小划分csv
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    audio_files = glob(path)
    if not audio_files:
        print(f"No audio files found in {path}")
        return
    # 获取音频文件列表和文件大小
    file_sizes = [(audio_file, os.path.getsize(audio_file)) for audio_file in audio_files]
    # 根据文件大小排序
    file_sizes.sort(key=lambda x: x[1])

    for i, (audio_file,size) in enumerate(file_sizes):
        df=pd.read_csv(audio_file)
        df['lable'] = df['lable'].fillna(0).astype(int)
        if i<4:
            test_df = pd.concat([df,test_df], axis=0, join='outer')
        else:
            train_df = pd.concat([df,train_df], axis=0, join='outer')
    print(f"Data has been splited to train and test") 
    return train_df,test_df 
  

  @classmethod
  def prepare_dataset(self,X_train, X_test, y_train, y_test):
    train_data = CustomDataset(X_train, y_train)
    test_data = CustomDataset(X_test, y_test)
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    return train_data, test_data, trainloader, testloader
  

  def LSTM_TRAIN(self,X_train, X_test, y_train, y_test, count):
    train_data, test_data,trainloader, testloader = self.prepare_dataset(X_train, X_test, y_train, y_test)
    lstm = LSTM()
    print(lstm)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
    steps = [] 
    accuracies = []
    for epoch in range(EPOCH):
      #训练模型
      for step, (b_x, b_y) in enumerate(trainloader):
          b_x = b_x.view(-1, 1, INPUT_SIZE).float()
          output = lstm(b_x)
          loss = criterion(output, b_y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      #测试模型
      sum_correct = 0
      y_true = []
      y_pred = []
      lable_correct={}
      lable_total={}
      with torch.no_grad():
          for step, (a_x, a_y) in enumerate(testloader):
              a_x = a_x.view(-1, 1, INPUT_SIZE)
              test_output = lstm(a_x)
              pred_y = torch.max(test_output, 1)[1]
              #print(f'pred_y: {pred_y}, a_y: {a_y}')
              y_true.extend(a_y.cpu().numpy())
              y_pred.extend(pred_y.cpu().numpy())
              sum_correct += (pred_y == a_y).sum().item()

              # 统计每个标签的正确预测数量和总预测数量
              #zip返回两个列表的对应元素
              for true_lable, pred_lable in zip(a_y.cpu().numpy(), pred_y.cpu().numpy()):
                  if true_lable not in lable_correct:#进行必要的初始化
                    lable_correct[true_lable] = 0
                    lable_total[true_lable] = 0
                  lable_total[true_lable] += 1
                  if true_lable == pred_lable:
                    lable_correct[true_lable] += 1
      # 计算每个标签的准确率
      lable_accuracy = {}
      for label in lable_correct:
          if lable_total[label] > 0:
              lable_accuracy[label] = lable_correct[label] / lable_total[label]
          else:
              lable_accuracy[label] = 0.0

      # 打印每个标签的准确率
      for label, accuracy in lable_accuracy.items():
          print(f'Label {label}: Accuracy {accuracy:.4f}')

    # 计算整体准确率
      sum_correct = sum(lable_correct.values())
      accuracy = sum_correct / len(test_data)
      steps.append(epoch)
      accuracies.append(accuracy)
      precision = precision_score(y_true, y_pred, average='weighted')
      f1 = f1_score(y_true, y_pred, average='weighted')
      recall = recall_score(y_true, y_pred, average='weighted')
      #一个大EPOCH只打印一次热力图
      if epoch==EPOCH-1:
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 按行归一化
        cm_percent = cm_normalized * 100
        sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Confusion Matrix (Percentage)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
      print(f'Epoch [{epoch + 1}/{EPOCH}], Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1 score: {f1:.4f}, Recall: {recall:.4f}, train loss: {loss.item():.4f}')

    plt.plot(steps, accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.legend()
    #plt.show()
    tt = time() - t0
    print("WHOLE time is {} seconds".format(round(tt, 3)))

  def plot_roc(self,steps, accuracies):
    plt.plot(steps, accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.legend()
    plt.show()

  def MLP_TRAIN(self,X_train, X_test, y_train, y_test, count):
      
    train_data, test_data,trainloader, testloader = self.prepare_dataset(X_train, X_test, y_train, y_test)
    
    mlp = MLP()
    print(mlp)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)

    steps = [] 
    accuracies = []
    for epoch in range(EPOCH):
      #训练模型
      for step, (b_x, b_y) in enumerate(trainloader):
          b_x = b_x.view(-1, INPUT_SIZE).float()
          output = mlp(b_x)
          loss = criterion(output, b_y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      #测试模型
      sum_correct = 0
      y_true = []
      y_pred = []
      lable_correct={}
      lable_total={}
      with torch.no_grad():
          for step, (a_x, a_y) in enumerate(testloader):
              a_x = a_x.view(-1, INPUT_SIZE)
              test_output = mlp(a_x)
              pred_y = torch.max(test_output, 1)[1]
              #print(f'pred_y: {pred_y}, a_y: {a_y}')
              y_true.extend(a_y.cpu().numpy())
              y_pred.extend(pred_y.cpu().numpy())
              sum_correct += (pred_y == a_y).sum().item()

              # 统计每个标签的正确预测数量和总预测数量
              # zip返回两个列表的对应元素
              for true_lable, pred_lable in zip(a_y.cpu().numpy(), pred_y.cpu().numpy()):
                  if true_lable not in lable_correct:#进行必要的初始化
                    lable_correct[true_lable] = 0
                    lable_total[true_lable] = 0
                  lable_total[true_lable] += 1
                  if true_lable == pred_lable:
                    lable_correct[true_lable] += 1
      # 计算每个标签的准确率
      lable_accuracy = {}
      for lable in lable_correct:
          if lable_total[lable] > 0:
              lable_accuracy[lable] = lable_correct[lable] / lable_total[lable]
          else:
              lable_accuracy[lable] = 0.0

      # 打印每个标签的准确率
      for lable, accuracy in lable_accuracy.items():
          print(f'Label {lable}: Accuracy {accuracy:.4f}')

    # 计算整体准确率
      sum_correct = sum(lable_correct.values())
      accuracy = sum_correct / len(test_data)
      steps.append(epoch)
      accuracies.append(accuracy)
      precision = precision_score(y_true, y_pred, average='weighted')
      f1 = f1_score(y_true, y_pred, average='weighted')
      recall = recall_score(y_true, y_pred, average='weighted')
      #一个大EPOCH只打印一次热力图
      if epoch==EPOCH-1:
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 按行归一化
        cm_percent = cm_normalized * 100
        sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Confusion Matrix (Percentage)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
      print(f'Epoch [{epoch + 1}/{EPOCH}], Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1 score: {f1:.4f}, Recall: {recall:.4f}, train loss: {loss.item():.4f}')

    plt.plot(steps, accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.legend()
    #plt.show()
    tt = time() - t0
    print("WHOLE time is {} seconds".format(round(tt, 3)))

  def DecisionTree(self,X_train,X_test,y_train,y_test):
      # 定义参数空间
      param_dist = {
          'criterion': ['gini', 'entropy'],
          'splitter': ['best', 'random'],
          'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
          'max_features': ['auto', 'sqrt', 'log2', None]
      }
      # 初始化RandomizedSearchCV
      clf = DecisionTreeClassifier(random_state=42)
      random_search = RandomizedSearchCV(estimator=clf, 
                                        param_distributions=param_dist, 
                                        n_iter=100, cv=5, verbose=2, 
                                        random_state=42, 
                                        n_jobs=-1)
      t0 = time()
      random_search.fit(X_train, y_train)
      tt = time() - t0
      print("Randomsearch in {} seconds".format(round(tt, 3)))
      t0 = time()
      clf= random_search.best_estimator_#使用最佳参数训练模型
      y_predict = clf.predict(X_test)
      tt = time() - t0
      print("Predicted in {} seconds".format(round(tt, 3)))
      # Showing Results 用函数来看预测结果的准确性
      accuracy = accuracy_score(y_test, y_predict)
      print("Accuracy is {}."  .format(round(accuracy, 4)))
      precision = precision_score(y_test, y_predict, average='weighted')
      print("Precision is {}.".format(round(precision, 4)))
      recall = recall_score(y_test, y_predict, average='weighted')
      print("Recall is {}.".format(round(recall, 4)))
      print("F is {}.".format(round(f1_score(y_test, y_predict, average='weighted'), 4)))

    
  def LSTM_Predict(self,X_train,X_test,y_train,y_test,count):
    EPOCH = 100              # train the training data n times
    BATCH_SIZE = 128
    TIME_STEP = None    # time step
    INPUT_SIZE = 1024        # input size   
    LR = 0.01               # learning rate
    seq_length = 3  # 定义输入序列的长度
    
    # 创建数据集和数据加载器
    
    train_dataset = CustomDataset(X_train, seq_length=seq_length)
    test_dataset = CustomDataset(X_test, seq_length=seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class LSTM(nn.Module):
      def __init__(self):
        super(LSTM, self).__init__()#super()调用父类,这个语句等同于nn.Module.
        self.rnn = nn.LSTM(         # 调用nn.LSTM
          input_size=1024,         # 输入数据的特征维度
          hidden_size=32,         # rnn hidden unit
          num_layers=1,           # number of rnn layer
          batch_first=True,       #改变形状e.g. (batch, time_step, input_size)
          )
        self.out = nn.Linear(32,1024)

        def forward(self, x):
            lstm_out, _ = self.rnn(x)
            out = self.out(lstm_out[:, -1, :])  # 取最后一个时间步的输出
            return out  # 压缩维度以匹配标签的形状
    lstm = LSTM()
    # 定义损失函数
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.NLLLoss()
    # 定义优化器
    optimizer_sgd = torch.optim.SGD(lstm.parameters(), lr=LR)
    optimizer_adam = torch.optim.Adam(lstm.parameters(), lr=LR)
    ''' #选择优化器和损失函数
    optimizer = optimizer_sgd  # optimize all cnn parameters
    loss_func = criterion1'''
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)


    #训练模型
    for epoch in range(EPOCH):
      for step, (b_x, b_y) in enumerate(train_dataloader):   # enumerate(dataloader)返回一个可迭代对象，其中每个元素是一个包含两个元素的元组，分别为step和(b_x, b_y)。其中，step表示当前迭代次数，b_x表示当前迭代所对应的输入数据，b_y表示当前迭代所对应的标签数据
        print(f'b_x shape is {b_x.shape}, b_y shape is {b_y.shape}')
        #print(f'b_x is  {b_x}, b_y is {b_y}')
        output = lstm(b_x)
        loss_func = criterion(output, b_y)
        optimizer.zero_grad()
        loss_func.backward()
        optimizer.step()  
      if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss_func.item():.4f}')                         
      # 在训练完成后，模型已经学习到了时间序列的模式
      # 现在，可以用测试数据集来预测下一个时间步的值
    lstm.eval()
      # 用于存储预测结果的列表
      # 预测后续时间步的值
    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, (a_x, a_y) in enumerate(test_dataloader):
            test_output = lstm(a_x)
            y_true.append(a_y.squeeze().numpy())  # 将真实值添加到列表中
            y_pred.append(test_output.squeeze().numpy())  # 将预测值添加到列表中
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
    
    # 计算准确率
    accuracy = accuracy_score(y_true.view(-1), torch.round(y_pred.view(-1)))
    print(f'Accuracy: {accuracy:.4f}')
'''        for i in range(len(X_test) - seq_length):
            input_seq = test_dataloader[i:i+seq_length].unsqueeze(0).unsqueeze(-1)  # 取当前序列作为输入
            predicted_value = lstm(input_seq).item()  # 使用模型预测下一个时间步的值
            predicted_values.append(predicted_value)
            # 将预测的值添加到测试数据中，用于后续预测下一个时间步
            test_data.append(predicted_value)'''

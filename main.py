import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
from time import time
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle
from Mutiple_Machine_learning import Mutiple_Methods as mm
from Mutiple_Machine_learning import LSTM 
from sklearn import preprocessing
from scipy.signal import medfilt
import threading
#将分帧后的音频数据传给train_test_split进行数据集的划分，导致了模型准确率过高，原因是否为分帧后的每一个数据都非常相似，因此相当于测试集和训练集的区别不大，导致预测无效
#如果数据非常的相似，不能合成一个进行后使用sklearn进行训练，不然会导致数据泄露，应当将文件分成训练集和测试集后，在进行数据处理


def get_trimmed_picture(audio_file):
    y, sr = librosa.load(audio_file)
    y_trimmed,_=librosa.effects.trim(y)#修剪掉部分的细节
    #画图环节
    sns.set_theme(style="white",palette=None)
    color_pal=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle=cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    pd.Series(y_trimmed).plot(figsize=(10,5),lw=1,
               title='Raw Audio Exampel',
                  color=color_pal[1])
    plt.show()
    return y_trimmed

def forget(y):
    D=librosa.stft(y)
    S_db=librosa.amplitude_to_db(np.abs(D),ref=np.max)
    print(S_db.shape)
    fig,ax=plt.subplots(figsieze=(10,5))
    img=librosa.display.specshow(S_db,x_axis='time',
                             y_axis='log',
                             ax=ax)
    ax.set_title('Spectogram Example',fontsize=20)
    fig.colorbar(img,ax=ax,format=f'%0.2f')
    '''# 提取梅尔频谱图作为特征
        mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, fmax=8000)
        y = librosa.power_to_db(mel_spectrogram)'''


def become_one(path1,path2,path3):#文件加起来太大了不容易变成一个
    previous_df = pd.DataFrame()
    for path in [path1,path2,path3]:
        audio_files = glob(path)
        if not audio_files:
            print(f"No audio files found in {path}")
            return
        for i, audio_file in enumerate(audio_files):
            df=pd.read_csv(audio_file)
            previous_df = pd.concat([df, previous_df], axis=0, join='outer')
    previous_df.to_csv('total_data.csv', index=False)  # 导出为 CSV 文件
    print(f"Data has been saved to total_data.csv") 
    return previous_df    
        
    

def pad_audio(y, frame_length, hop_length):
    # 计算需要填充的长度
    pad_length = frame_length - len(y) % hop_length
    if pad_length == frame_length:
        pad_length = 0
    # 填充音频信号
    y_padded = np.pad(y, (0, pad_length), mode='constant')
    return y_padded

def read_audio(audio_file, frame_length, hop_length):
    y, sr = librosa.load(audio_file)
    y, _ = librosa.effects.trim(y)# y,_=librosa.effects.trim(y)#修剪掉部分的细节
    y=np.pad(y,(0,5000),mode='constant')
    #y = pad_audio(y, frame_length, hop_length)
    return y,sr


def become_data(path, name,SF,id):
    Frame_Length=4096
    Hop_Length=400
    audio_files = glob(path)
    if not audio_files:
        print(f"No audio files found in {path}")
        return
    for i, audio_file in enumerate(audio_files):
        y,sr=read_audio(audio_file,Frame_Length,Hop_Length)
        frames = librosa.util.frame(y, frame_length=Frame_Length, hop_length=Hop_Length).T
        haming_window=np.hamming(Frame_Length)
        frames=frames*haming_window
        #y = medfilt(y, 3)  # 中值滤波
        # frame_size:帧大小 hop_length: 帧移 
        
        # 对每一帧进行傅里叶变换
        frames = np.abs(np.fft.fft(frames, axis=1))
        print(f'audio_file is {audio_file}')
        print(f'frame_size is {frames.shape[0]}')
        #frames.shape[0]是帧数
        previous_df_frame = pd.DataFrame()#一个文件生成一个dataframe，因此放在里面
        for j in range(frames.shape[0]):
            frame = frames[j]
            df_frame = pd.DataFrame({j+1: frame})  # 创建一个 DataFrame，其中列名为音频文件的索引
            previous_df_frame = pd.concat([previous_df_frame, df_frame], axis=1, join='outer')
            #print(f'previous_df_frame is {previous_df_frame}')
        df = previous_df_frame.transpose()  # 转置 DataFrame  
        df=df.fillna(0,axis=1)#去除NA
        df=df.astype('float64')#转为为float64
        '''scaler = preprocessing.MinMaxScaler()#将特征向量标准化
        df = pd.DataFrame(scaler.fit_transform(df))'''
        #将lable转为数字
        df_lable=0
        if name=='hengming':
            df_lable=1
        elif name=='tuqi':
            df_lable=2
        elif name=='xiqi':
            df_lable=3
        #根据SF里面来添加标签，转置后放在最后一列
        df_lable=pd.DataFrame({'lable':[df_lable if x==1 else 0 for x in SF]})
        df=pd.concat([df,df_lable],axis=1) 
        if name=='hengming':
            filepath = r'D:\桌面\sound_analysis\hengming_fft\{}{}.csv'.format(name, id)
        elif name=='tuqi':
            filepath = r'D:\桌面\sound_analysis\tuqi_fft\{}{}.csv'.format(name, id)
        elif name=='xiqi':
            filepath = r'D:\桌面\sound_analysis\xiqi_fft\{}{}.csv'.format(name, id)
        df=df.fillna(0,axis=1)#去除NA
        df.to_csv(filepath, index=False)  # 导出为 CSV 文件
        print(f"Data has been saved to {filepath}") 


def inplace_feature(path,name='unknown'):
    files = glob(path)
    id=0#音频文件索引
    for i, audio_file in enumerate(files):
        id=i+1
        df=pd.read_csv(audio_file)
        df=extract_feature(df,name+str(id))
        df.to_csv(audio_file,index=False)
def extract_feature(df,name='unknown'):
    columns = ['sig_mean', 'sig_std', 'rmse_mean', 'rmse_std', 'silence', 'harmonic', 'auto_corr_max', 'auto_corr_std','lable']
    df_features = pd.DataFrame(columns=columns)
    for i in range(0,len(df)):
        try:
            df_row = df.iloc[i].values.flatten().astype('float32')# 保证df_row是一维数组
            # 对输入信号进行填充 n_fft: 傅里叶变换的窗口大小必须小于等于信号长度
            n_fft = 2048
            if len(df_row) < n_fft:
                df_row = np.pad(df_row, (0, n_fft - len(df_row)), mode='constant')
            
            rmse = librosa.feature.rms(df_row + 0.0001)[0]#均方根差
            silence = 0
            for e in rmse:
                if e <= 0.4 * np.mean(rmse):
                    silence += 1
            y_harmonic, y_percussive = librosa.effects.hpss(df_row)
            autocorr = librosa.core.autocorrelate(df_row)
            cl = 0.45 * np.mean(abs(df_row))
            center_clipped = []
            for s in df_row:
                if s >= cl:
                    center_clipped.append(s - cl)
                elif s <= -cl:
                    center_clipped.append(s + cl)
                elif np.abs(s) < cl:
                    center_clipped.append(0)
            new_autocorr = librosa.core.autocorrelate(np.array(center_clipped))
            df_new_row=[]
            sig_mean=np.mean(abs(df_row))
            df_new_row.append(sig_mean)  # sig_mean
            df_new_row.append(np.std(df_row))  # sig_std
            df_new_row.append(np.mean(rmse))  # rmse_mean
            df_new_row.append(np.std(rmse))  # rmse_std
            df_new_row.append(silence)  # silence
            df_new_row.append(np.mean(y_harmonic) * 1000)  # harmonic (scaled by 1000)
            df_new_row.append(1000 * np.max(new_autocorr)/len(new_autocorr))  # auto_corr_max (scaled by 1000)
            df_new_row.append(np.std(new_autocorr))  # auto_corr_std
            if name=='hengming':
                df_new_row.append(0)
            elif name=='tuqi':
                df_new_row.append(1)
            elif name=='xiqi':
                df_new_row.append(2)
            df_features = df_features.append(pd.DataFrame(df_new_row, index=columns).transpose(), ignore_index=True)
            #print(f'df_features is {df_features}')
            df_features=df_features.fillna(0,axis=1)#填充NA为0
        except Exception as e:
            print('Some exception occurred: {}'.format(e))
    '''print(f'df_features is {df_features}')
    print(f'df_features shape is{df_features.shape}')'''
    return df_features

def get_T1_T2(audio_file):
    IS = 0.25  # 静音段的长度，单位是秒
    wnd = 4096  # 帧大小
    inc = 400  # 帧移
    thr1 = 0.99  # 阈值1
    thr2 = 0.96  # 阈值2
    wlen = 4096  # 窗口长度
    y, sr = read_audio(audio_file, wnd, inc)
    NIS = int((IS * sr - wlen) // inc + 1)  # 计算静音段帧数
    # 将音频信号分帧，帧大小为 wnd，帧移为 inc，结果是一个二维数组，每行是一个帧
    frames = librosa.util.frame(y, frame_length=wnd, hop_length=inc).T
    # 对每一帧进行傅里叶变换
    frames = np.abs(np.fft.fft(frames, axis=1))
    # 计算频率分辨率
    df = sr / wlen
    fx1 = int(250 // df + 1)  # 250Hz 位置
    fx2 = int(3500 // df + 1)  # 3500Hz 位置
    km = wlen // 8
    K = 0.5  # 一个常数
    # 初始化能量矩阵
    E = np.zeros((frames.shape[0], wlen // 2))
    # 提取 250Hz 到 3500Hz 之间的频率分量
    E[:, fx1 + 1:fx2 - 1] = frames[:, fx1 + 1:fx2 - 1]
    # 将每个频率分量平方，计算能量
    E = np.multiply(E, E)
    # 计算每帧的总能量
    Esum = np.sum(E, axis=1, keepdims=True)
    # 计算能量分布比例
    P1 = np.divide(E, Esum)
    # 将能量分布比例大于等于 0.9 的频率分量置零
    E = np.where(P1 >= 0.9, 0, E)
    # 将频率分量分组，并计算每组的总能量
    Eb0 = E[:, 0::4]
    Eb1 = E[:, 1::4]
    Eb2 = E[:, 2::4]
    Eb3 = E[:, 3::4]
    Eb = Eb0 + Eb1 + Eb2 + Eb3
    # 计算每组的概率分布
    prob = np.divide(Eb + K, np.sum(Eb + K, axis=1, keepdims=True))
    # 计算每帧的熵
    Hb = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
    # 对熵进行平滑处理
    for i in range(10):
        Hb = medfilt(Hb, 5)
    # 计算平均熵
    Me = np.mean(Hb)
    # 计算静音段的平均熵
    eth = np.mean(Hb[:NIS])
    # 计算熵的差值
    Det = eth - Me
    # 计算阈值 T1 和 T2
    T1 = thr1 * Det + Me
    T2 = thr2 * Det + Me
    # 打印阈值和熵的形状
    print(f'T1 is {T1}, T2 is {T2}')
    #print(f'Hb is {Hb}')
    print(f'Hb shape is {Hb.shape}')
    # 调用 find_end_point 函数找到音频的端点
    SF, NF = find_end_point(Hb, T1, T2)
    return y, sr, Hb, SF, NF
     

def find_end_point(y, T1, T2):
    y_length = len(y)
    maxsilence = 8
    minlen = 5
    status = 0
    audio_length = np.zeros(y_length)
    audio_silence = np.zeros(y_length)
    segment_id = 0
    audio_start = np.zeros(y_length)
    audio_finish = np.zeros(y_length)
    for n in range(1, y_length):
        if status == 0 or status == 1:
            if y[n] < T2:
                audio_start[segment_id] = max(1, n - audio_length[segment_id] - 1)
                status = 2
                audio_silence[segment_id] = 0
                audio_length[segment_id] += 1
            elif y[n] < T1:
                status = 1
                audio_length[segment_id] += 1
            else:
                status = 0
                audio_length[segment_id] = 0
                audio_start[segment_id] = 0
                audio_finish[segment_id] = 0
        if status == 2:
            if y[n] < T1:
                audio_length[segment_id] += 1
            else:
                audio_silence[segment_id] += 1
                if audio_silence[segment_id] < maxsilence:
                    audio_length[segment_id] += 1
                elif audio_length[segment_id] < minlen:
                    status = 0
                    audio_silence[segment_id] = 0
                    audio_length[segment_id] = 0
                else:
                    status = 3
                    audio_finish[segment_id] = audio_start[segment_id] + audio_length[segment_id]
        if status == 3:
            status = 0
            segment_id += 1
            audio_length[segment_id] = 0
            audio_silence[segment_id] = 0
            audio_start[segment_id] = 0
            audio_finish[segment_id] = 0
    segment_num = len(audio_start[:segment_id])
    if audio_start[segment_num - 1] == 0:
        segment_num -= 1
    if audio_finish[segment_num - 1] == 0:
        print('Error: Not find ending point!\n')
        audio_finish[segment_num] = y_length
    SF = np.zeros(y_length)
    NF = np.ones(y_length)
    for i in range(segment_num):
        SF[int(audio_start[i]):int(audio_finish[i])] = 1
        NF[int(audio_start[i]):int(audio_finish[i])] = 0
    return SF, NF


def plot_audio_and_classification(audio_file):
    y, sr, Hb, SF, NF = get_T1_T2(audio_file)
    
    # 计算时间轴
    time = np.arange(len(y)) / sr
    frame_time = np.arange(len(Hb)) * (1024 / sr)
    # 创建图形
    plt.figure(figsize=(12, 8))
    # 绘制音频信号
    plt.subplot(3, 1, 1)
    plt.plot(time, y)
    plt.title('Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # 绘制熵值
    plt.subplot(3, 1, 2)
    plt.plot(frame_time, Hb)
    plt.title('Entropy')
    plt.xlabel('Time (s)')
    plt.ylabel('Entropy')
    # 绘制分类结果
    plt.subplot(3, 1, 3)
    plt.plot(frame_time, SF, label='Speech')
    plt.plot(frame_time, NF, label='Non-Speech')
    plt.title('Classification Result')
    plt.xlabel('Time (s)')
    plt.ylabel('Classification')
    plt.legend()
    plt.tight_layout()
    plt.show()

def prepare_data(path,name):
    audio_files = glob(path)
    id=0
    if not audio_files:
        print(f"No audio files found in {path}")
        return
    for i, audio_file in enumerate(audio_files):
        id+=1
        y, sr, Hb, SF, NF=get_T1_T2(audio_file)
        become_data(audio_file,name,SF,id)


def read_and_train_data(path):
    audio_files = glob(path)
    if not audio_files:
        print(f"No audio files found in {path}")
        return
    for i, audio_file in enumerate(audio_files):
        df=pd.read_csv(audio_file)
        df=df[df['lable']!=0]
        X_train,X_test,y_train,y_test=mm.split_data_with_sklearn(df)
        mm.LSTM_TRAIN(X_train,X_test,y_train,y_test,len(df['lable'].value_counts()))

def read_and_train_data_with_better_split(path1,path2,path3):
    train_df1,test_df1=mm.split_data_with_size(path1)
    train_df2,test_df2=mm.split_data_with_size(path2)
    train_df3,test_df3=mm.split_data_with_size(path3)
    train_df=pd.concat([train_df1,train_df2,train_df3])
    test_df=pd.concat([test_df1,test_df2,test_df3])
    train_df=train_df[train_df['lable']!=0]
    test_df=test_df[test_df['lable']!=0]
    '''train_df=train_df[:-1]#撇掉最后一行，因为最后一行是空的
    test_df=test_df[:-1]'''
    mm.LSTM_TRAIN(train_df.iloc[:,0:-1],test_df.iloc[:,0:-1],train_df['lable'],test_df['lable'],len(train_df['lable'].value_counts()))

def conbine_land_water_and_train(water_path,land_path): 
    train_df=pd.DataFrame() 
    test_df=pd.DataFrame() 
    for i,j in zip(land_path,water_path):
        train_df1,test_df1=mm.split_data_with_size(i)
        train_df2,test_df2=mm.split_data_with_size(j)
        train_df=pd.concat([train_df1,train_df2,train_df])
        test_df=pd.concat([test_df1,test_df2,test_df])
    train_df=train_df[train_df['lable']!=0]
    test_df=test_df[test_df['lable']!=0]
    mm.LSTM_TRAIN(train_df.iloc[:,0:-1],test_df.iloc[:,0:-1],train_df['lable'],test_df['lable'],len(train_df['lable'].value_counts()))

if __name__ == '__main__':
    # 使用args参数
    thread1 = threading.Thread(target=prepare_data, args=(r'tuqi/*.mp3', 'tuqi'), daemon=True)
    thread1.start()
    thread2 = threading.Thread(target=prepare_data, args=(r'xiqi/*.mp3', 'xiqi'), daemon=True)
    thread2.start()
    thread3 = threading.Thread(target=prepare_data, args=(r'hengming/*.mp3', 'hengming'), daemon=True)
    thread3.start()
    threads=[thread1,thread2,thread3]
    for t in threads:
        t.join()
    # 等待三个线程结束
    '''df=become_one(r'hengming_fft/*.csv',r'tuqi_fft/*.csv',r'xiqi_fft/*.csv')
    X_train,X_test,y_train,y_test=mm.split_data(df) 
    mm.LSTM_TRAIN(X_train,X_test,y_train,y_test,len(df['lable'].value_counts()))'''
    #read_and_train_data(r'total_data.csv')
    #read_and_train_data_with_better_split(r'hengming_fft_no01/*.csv',r'tuqi_fft_no01/*.csv',r'xiqi_fft_no01/*.csv')
    read_and_train_data_with_better_split(r'hengming_fft/*.csv',r'tuqi_fft/*.csv',r'xiqi_fft/*.csv')
    #conbine_land_water_and_train([r'hengming_fft_no01/*.csv',r'tuqi_fft_no01/*.csv',r'xiqi_fft_no01/*.csv']
                                 #,[r'hengming_land_fft/*.csv',r'tuqi_land_fft/*.csv',r'xiqi_land_fft/*.csv'])
    #plot_audio_and_classification(r'hengming_land/24_07_14_15_02_14.wav')
    
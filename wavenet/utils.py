import numpy as np

from scipy.io import wavfile
import os
import matplotlib.pyplot as plt


import urllib.request
import json
import numpy as np


class BitCoinDataset():
    
    def __init__(self, num_time_steps):
        print("init of bitcoin dataset")

        self.num_time_steps = num_time_steps

        #f = open("dataset/histohour1.json", 'r')
        #self.histohour1 = json.load(f)
        #f = open("dataset/histohour2.json", 'r')
        #self.histohour2 = json.load(f)
        #self.histohour1 = self.get_json(2000, "&toTs=1521370800")
        #self.histohour2 = self.get_json(2000, "&toTs=1514170800")
        self.histohour = np.load("dataset/normalized_histohour.npy")
        
    def len(self):
        print("length")

    # 指定されたタイムステップで時間足を取得。
    def get_json(self, num_time_steps, toTs):
        #rand_toTs = np.random.randint(low=1506970800 + num_time_steps, high=1521370800)
        #print(rand_toTs)
        url = 'https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=JPY&limit={0}&e=bitFlyer{1}'.format(num_time_steps-1, toTs)
        #print(url)
        response = urllib.request.urlopen(url)
        content = json.loads(response.read().decode('utf8'))
        return content

    # batch内の1 dataを作成
    def make_single_batch(self):
        start = np.random.randint(self.histohour.shape[0] - (self.num_time_steps+1))
        return self.histohour[start:start + self.num_time_steps+1]
        """
        rand = np.random.randint(10) % 2
        if rand == 0:
            hours_content = self.histohour1
        else:
            hours_content = self.histohour2

        #hours_content = self.get_json(num_time_steps)
        #print(type(hours_content["Data"]))
        #print(len(hours_content["Data"]))

        start = np.random.randint(len(hours_content["Data"]) - (self.num_time_steps+1))
        
        single_batch = []
        for i in range(self.num_time_steps+1):
            data = hours_content["Data"][start + i]
            tmp_data = [data["close"], data["high"], data["low"], data["open"], data["volumefrom"], data["volumeto"]]
            single_batch.append(tmp_data)
            #print(data["time"])

        return single_batch
        """
        
    # 正規化
    def min_max_normalize(self, x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return result
    
    # batchを作成
    def make_multi_batch(self, num_batch_size, num_time_steps):
        multi_batch = []
        for i_batch in range(num_batch_size):
            multi_batch.append(self.make_single_batch())
        
        # 正規化
        norm_multi_batch_np = np.array(multi_batch)
        #norm_multi_batch_np = self.min_max_normalize(multi_batch_np, axis=1)*2.0 -1.0

        batch_input_np = norm_multi_batch_np[:,:-1,:]
        #print(batch_input_np.shape)
        batch_output_np = norm_multi_batch_np[:,1:,:]
        #batch_output_np = batch_output_np[:,np.newaxis]
        
        return batch_input_np, batch_output_np

    def make_test_batch(self):
        # get latest data
        hours_content = self.get_json(self.num_time_steps, "")
        
        #start = np.random.randint(len(hours_content["Data"]) - (self.num_time_steps+1))
        single_batch = []
        for data in hours_content["Data"]:
            #data = hours_content["Data"][start + i]
            tmp_data = [data["close"], data["high"], data["low"], data["open"], data["volumefrom"], data["volumeto"]]
            single_batch.append(tmp_data)
        
        single_batch_np = np.array(single_batch)
        #norm_single_batch_np = self.min_max_normalize(single_batch_np, axis=1)*2.0 -1.0

        return single_batch_np
    
    """
    def getSequentialItem(self, time_step, batch_size=2):
        #rand_batch_idx = np.random.randint(0, self.len()-1, size=batch_size)
        #print(rand_batch_idx)
        batch_idx = np.arange(batch_size)
        
        batch_input = []
        batch_output = []

        item = self.getitem(0)
        time_series = item["wave"].shape[1]-batch_size
        #print(time_step)
        #print(batch_size)
        #print(time_series)
        for time in np.arange(batch_size):
            time_start = 0
            
            wave_time_series = item["wave"][3,time_start + time:time_start+time+time_step][np.newaxis,:].transpose()
            one_hot = item["label_one_hot"]
            one_hot_time_series = np.repeat(one_hot, time_step, axis=0)

            input = np.concatenate((wave_time_series, one_hot_time_series), axis=1)
            #print(input.shape)
            batch_input.append(input)
            batch_output.append(item["wave"][2,time_start+time+time_step].transpose())
            
        batch_input_np = np.array(batch_input)
        batch_output_np = np.array(batch_output)
        return batch_input_np, batch_output_np[:, np.newaxis]
    """

def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path):
    data = wavfile.read(path)[1][:, 0]

    data_ = normalize(data)
    # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    bins = np.linspace(-1, 1, 256)
    #print(data_[0:-1].shape)
    #print(data_[1::].shape)
    # Quantize inputs.
    inputs = np.digitize(data_[0:-1], bins, right=False) - 1
    inputs = bins[inputs][None, :, None]

    # Encode targets as ints.
    targets = (np.digitize(data_[1::], bins, right=False) - 1)[None, :]
    return inputs, targets

def one_hot_encoding(t_vec, num_classes):
    #t_oh = np.zeros((len(t_vec), num_classes)).astype(int)
    #t_oh[np.arange(len(t_vec)), t_vec] = 1
    #return t_oh
    targets = np.array(t_vec).reshape(-1)
    t_oh = np.eye(num_classes)[targets].astype("int")
    return (t_oh)

Selected_Acceleration_Label = {
    "G1SquaredAluminumMesh":0,
    "G2GraniteTypeVeneziano":1,
    "G3AluminumPlate":2,
    "G4Bamboo":3,
    "G5SolidRubberPlateVersion1":4,
    "G6Carpet":5,
    "G7FineFoamVersion2":6,
    "G8Cardboard":7,
    "G9Jeans":8,
    }    


class AccelerationDataset():
    
    def __init__(self, root_dir, name, duration_for_test):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)
        self.files_z = [file for file in self.files if (file.find("_Z_") > 0) and (file.find("acc") > 0) and (name in file)]
        self.duration_for_test = duration_for_test
        print(len(self.files_z))
        #print(self.files_z)
        
    def len(self):
        return len(self.files_z)
        #return 1
    

    def getitem(self, idx):
        # wave
        file_name_acc_z = self.files_z[idx][:-7]
        pos_Z_ = file_name_acc_z.find("_Z_")

        file_name_vel_x = file_name_acc_z[:pos_Z_] + "_X_" + file_name_acc_z[pos_Z_ + len("_Z_"):]
        file_name_vel_y = file_name_acc_z[:pos_Z_] + "_Y_" + file_name_acc_z[pos_Z_ + len("_Z_"):]
        file_name_vel_mag = file_name_acc_z[:pos_Z_] + "_mag_" + file_name_acc_z[pos_Z_ + len("_Z_"):]

        z = np.load(os.path.join(self.root_dir, file_name_acc_z + "acc.npy")) * 2.0 - 1.0
        x = np.load(os.path.join(self.root_dir, file_name_vel_x + "vel.npy"))
        y = np.load(os.path.join(self.root_dir, file_name_vel_y + "vel.npy"))
        mag_vel = np.load(os.path.join(self.root_dir, file_name_vel_mag + "vel.npy"))

        #stacked = np.stack((x,y))
        #norm_vel = np.linalg.norm(np.stack((x,y)), axis=0)
        #print(norm_vel.shape)
        wave = np.stack((x,y,z,mag_vel))

        # label
        pos_label = file_name_acc_z.find("_Movement_")
        label_name = file_name_acc_z[:pos_label]
        label_num = Selected_Acceleration_Label[label_name]
        label_one_hot = one_hot_encoding(label_num, 9)
        
        sample = {"wave": wave, "label_num": label_num, "label_name": label_name, "label_one_hot": label_one_hot}
        return sample

    
    def getBatchTrain(self, is_random, time_step, batch_size):
        if is_random == True:
            rand_batch_idx = np.random.randint(0, self.len(), size=batch_size)
        else:
            rand_batch_idx = np.arange(0, self.len())
        #print(rand_batch_idx)

        batch_input = []
        batch_output = []
        for idx in rand_batch_idx:
            item = self.getitem(idx)
            if is_random == True:
                rand_time_start = np.random.randint(0, item["wave"].shape[1] - time_step -1 - self.duration_for_test)
            else:
                rand_time_start = item["wave"].shape[1] - self.duration_for_test - 1
                
            #print(item["wave"].shape)
            wave_time_series = item["wave"][3,rand_time_start:rand_time_start+time_step][np.newaxis,:].transpose()
            one_hot = item["label_one_hot"]
            one_hot_time_series = np.repeat(one_hot, time_step, axis=0)
            #print(one_hot_time_series)
            
            input = np.concatenate((wave_time_series, one_hot_time_series), axis=1)
            #print(input.shape)
            batch_input.append(input)
            batch_output.append(item["wave"][2,rand_time_start+1: rand_time_start+time_step+1].transpose())
            
        batch_input_np = np.array(batch_input)
        batch_output_np = np.array(batch_output)
        return batch_input_np, batch_output_np#[:, np.newaxis]

    
    def getSequentialItem(self, time_step, batch_size=2):
        #rand_batch_idx = np.random.randint(0, self.len()-1, size=batch_size)
        #print(rand_batch_idx)
        batch_idx = np.arange(batch_size)
        
        batch_input = []
        batch_output = []

        item = self.getitem(0)
        time_series = item["wave"].shape[1]-batch_size
        #print(time_step)
        #print(batch_size)
        #print(time_series)
        for time in np.arange(batch_size):
            time_start = 0
            
            wave_time_series = item["wave"][3,time_start + time:time_start+time+time_step][np.newaxis,:].transpose()
            one_hot = item["label_one_hot"]
            one_hot_time_series = np.repeat(one_hot, time_step, axis=0)

            input = np.concatenate((wave_time_series, one_hot_time_series), axis=1)
            #print(input.shape)
            batch_input.append(input)
            batch_output.append(item["wave"][2,time_start+time+time_step].transpose())
            
        batch_input_np = np.array(batch_input)
        batch_output_np = np.array(batch_output)
        return batch_input_np, batch_output_np[:, np.newaxis]
    
def show_wave(wave, dirname, filename, y_lim=0):
    if os.path.isdir(dirname) == False:
        os.mkdir(dirname)
    
    plt.figure(figsize=(30,10))
    plt.title('wave files', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if y_lim != 0:
        plt.ylim(0, y_lim)
    plt.plot(wave, color='r',linewidth=1.0, alpha=0.5)
    plt.savefig(os.path.join(dirname,filename + '.png'))
    plt.close()

def show_test_wav(waves, dirname, filename, y_lim=0):
    if os.path.isdir(dirname) == False:
        os.mkdir(dirname)
    
    plt.figure(figsize=(30,10))
    plt.title('wave files', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if y_lim != 0:
        plt.ylim(0, y_lim)
    
    plt.plot(waves["test"], color='r',linewidth=1.0, label="test", alpha=0.3)
    plt.plot(waves["generated"], color='b', linewidth=1.0, label="generated",alpha=0.3)

    plt.legend()
    plt.savefig(os.path.join(dirname, filename + '.png'))
    plt.close()

    
def show_bc_transition(waves, dirname, filename, y_lim=0):
    if os.path.isdir(dirname) == False:
        os.mkdir(dirname)
    
    plt.figure(figsize=(30,10))
    plt.title('wave files', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if y_lim != 0:
        plt.ylim(0, y_lim)
    
    plt.plot(waves[:,1], color='r',linewidth=1.0, label="low", alpha=0.3)
    plt.plot(waves[:,2], color='b', linewidth=1.0, label="high",alpha=0.3)

    plt.legend()
    plt.savefig(os.path.join(dirname, filename + '.png'))
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from wavenet.layers import (_causal_linear, _output_linear, conv1d,
                    dilated_conv1d)
from wavenet.utils import *
import os
import numpy as np

class Model(object):
    def __init__(self,
                 num_time_samples=10000,
                 num_channels=6,
                 num_classes=512,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 batch_size=2,
                 gpu_num = "0",
                 model_name = "default_name"):
        
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gpu_num = gpu_num
        self.model_name = model_name
        self.batch_size = batch_size

        self.iter_save_fig = 5000
        self.iter_save_param = 10000
        self.iter_calc_loss = 2500
        self.iter_end_training = 500000
        
        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples, num_channels))

        h = inputs
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
                hs.append(h)

        
        outputs = conv1d(h,
                         num_classes*num_channels,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)

        print(outputs.shape)
        print(targets.shape)
        costs = 0
        for i in range(num_channels):
            #costs += tf.nn.sparse_softmax_cross_entropy_with_logits(
            #    logits=outputs[i*num_classes:(i+1)*num_classes], labels=targets[:,:,i])
            costs += tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=outputs[:,:,i*num_classes:(i+1)*num_classes], labels=targets[:,:,i])
        cost = tf.reduce_mean(costs)

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        self.config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=self.gpu_num
            )
        )
        sess = tf.Session(config=self.config)
        self.saver = tf.train.Saver(max_to_keep=3)
            
        ckpt = tf.train.get_checkpoint_state("./ckpt_" + self.model_name + "/")
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            self.saver.restore(sess, last_model)
            print("load: "+last_model)
        else:
            print("init")
            sess.run(tf.global_variables_initializer())
        self.inputs_ph = inputs
        self.targets_ph = targets
        self.outputs = outputs
        self.hs = hs
        self.costs = costs
        self.cost = cost
        self.train_step = train_step
        self.sess = sess
        
        # dataset
        self.bc_train = BitCoinDataset(self.num_time_samples)
        self.param_min = np.load("dataset/param_min.npy")
        self.param_max = np.load("dataset/param_max.npy")
        #self.ad_train = AccelerationDataset("acc_dataset_selected/train", "train10", 0)
        #self.ad_test = AccelerationDataset("acc_dataset_selected/test", "test10", self.num_time_samples)
                
        self.generate_init()
        self.count = 0


        
        """
        #print(self.ad_train.shape)
        #print(self.ad_test.shape)
        batch_x, batch_y = self.ad_test.getBatchTrain(False, self.num_time_samples, self.batch_size)
        batch_x = batch_x.reshape((-1, self.num_time_samples, 10))#self.num_input))        
        batch_x_0 = batch_x[:,:,0]#.reshape((self.batch_size, self.num_time_samples,-1))
        bins = np.linspace(-1, 1, 256)
        #print(batch_x.shape)
        #print(batch_y.shape)
        # Quantize inputs.
                
        inputs_batch, targets_batch = [],[]
        for batch in range(batch_x_0.shape[0]):
            x = batch_x_0[batch]
            y = batch_y[batch]
            inputs = np.digitize(x, bins, right=False) - 1
            inputs = bins[inputs][None, :, None]
            inputs_batch.append(inputs)
        
            # Encode targets as ints.
            targets = (np.digitize(y, bins, right=False) - 1)[None, :]
            targets_batch.append(targets)
            
        inputs_batch = np.vstack(inputs_batch)
        inputs_batch = np.concatenate((inputs_batch, batch_x[:,:,1:]), axis=2)
        targets_batch = np.vstack(targets_batch)

        self.test_inputs = inputs_batch
        self.test_targets = np.array(bins[targets_batch])
        #print(self.test_inputs.shape)
        #print(self.test_targets.shape)
        """
        
    def _train(self):
        batch_x, batch_y = self.bc_train.make_multi_batch(self.batch_size, self.num_time_samples)
        #print(batch_x.shape)
        #print(batch_y.shape)
        bins = np.linspace(-1, 1, self.num_classes)
        # Quantize inputs.

        inputs_batch, targets_batch = [],[]
        for batch in range(self.batch_size):
            x = batch_x[batch]
            y = batch_y[batch]
            inputs = np.digitize(x, bins, right=False) - 1
            inputs = bins[inputs]#[None, :, None]
            inputs_batch.append(inputs)
        
            # Encode targets as ints.
            targets = (np.digitize(y, bins, right=False) - 1)#[None, :]
            targets_batch.append(targets)

        
        inputs_batch = np.array(inputs_batch)
        targets_batch = np.array(targets_batch)

        feed_dict = {self.inputs_ph: inputs_batch, self.targets_ph: targets_batch}
        cost, _ = self.sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        return cost

    def train(self):
        losses = []
        terminal = False
        i = 0
        while not terminal:
            i += 1
            cost = self._train()

            if i % 100 == 0:
                print("i:{0}, cost:{1}".format(i, cost))
                losses.append(cost)
            
            if i > self.iter_end_training:
                path = "ckpt_" + self.model_name
                if not os.path.isdir(path):
                    os.makedirs(path)
                self.saver.save(self.sess, path + "/" + str(i) + ".ckpt")       
                terminal = True

            """
            if i % self.iter_calc_loss:
                num_block = self.test_inputs.shape[0]//2
                loss_local = []
                for j in range(num_block):
                    feed_dict = {self.inputs_ph: self.test_inputs[j*2:(j+1)*2], self.targets_ph: self.test_targets[j*2:(j+1)*2]}
                    cost = self.sess.run(self.cost, feed_dict=feed_dict)
                    loss_local.append(cost)

                cost_mean = np.mean(loss_local)
                losses.append(cost_mean)

                if cost_mean < 1e-2:
                    path = "ckpt_" + self.model_name
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    self.saver.save(self.sess, path + "/" + str(i) + ".ckpt")       
                    terminal = True
            """
            
   
            if i % self.iter_save_fig == 0:
                print("")
                #train_generated = self.generate_run(self.train_inputs[0,:,:][np.newaxis,:,:], i)
                #waves = {"test":self.train_targets[0] , "generated":train_generated[0,:]}
                #show_test_wav(waves, dirname=self.model_name, filename="train_wave_" + str(i))
                """
                test_input = self.bc_train.make_test_batch()
                test_input_norm = 2.0*(test_input - self.param_min)/(self.param_max - self.param_min) -1.0
                #print(test_input_norm)
                
                test_generated_norm = self.generate_run(test_input_norm)
                test_generated = (test_generated_norm+1.0)*0.5
                test_generated = test_generated*(self.param_max-self.param_min) + self.param_min
                print(test_generated.shape)
                show_bc_transition(test_generated_norm, dirname=self.model_name, filename="norm_bc_transition_" + str(i))
                #waves = {"test":self.test_targets[0] , "generated":test_generated[0,:]}
                show_bc_transition(test_generated, dirname=self.model_name, filename="bc_transition_" + str(i))
                
                show_wave(losses, dirname=self.model_name, filename="losses_" + str(i), y_lim=6)
                """

                
            
            if i % self.iter_save_param == 0:
                path = "ckpt_" + self.model_name
                if not os.path.isdir(path):
                    os.makedirs(path)
                self.saver.save(self.sess, path + "/" + str(i) + ".ckpt")
            
                
    def generate_init(self, batch_size=1):
        self.bins = np.linspace(-1, 1, self.num_classes)
        input_size = self.num_channels #self.num_time_samples
        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print('Make Generator.')

        count = 0
        h = inputs

        init_ops = []
        push_ops = []
        for b in range(self.num_blocks):
            for i in range(self.num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = self.num_channels
                else:
                    state_size = self.num_hidden
                    
                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

                state_ = q.dequeue()
                push = q.enqueue([h])
                init_ops.append(init)
                push_ops.append(push)

                h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
                count += 1

        outputs = _output_linear(h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops
        
        # Initialize queues.
        self.sess.run(self.init_ops)

    def generate_run(self, inputs):
        # (200,6)
        predictions = []
        inputs_list = inputs.tolist()
        
        for i in range(inputs.shape[0]+100):
            inputs = np.array(inputs_list)

            feed_dict = {self.inputs: inputs[np.newaxis ,0,:]}
            output = self.sess.run(self.out_ops, feed_dict=feed_dict)[0] # ignore push ops
            #print(output.shape)

            values = []
            for i in range(self.num_channels):
                value = np.argmax(output[0, i*self.num_channels:(i+1)*self.num_channels])
                value_binned = np.array(self.bins[value])[None, None]
                values.append(np.squeeze(value_binned))

            values_np = np.array(values)
            #print(values_np.shape)
            #print(values_np)
            inputs_list.pop(0) # remove top
            inputs_list.append(values_np)
            
            #print("{0}: {1}".format(i, value_binned))
            
            predictions.append(values_np)

        return np.array(predictions[inputs.shape[0]-1:])
        
        
"""
class Generator(object):
    def __init__(self, model, batch_size=1, input_size=1):
        self.model = model
        self.bins = np.linspace(-1, 1, self.model.num_classes)

        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print('Make Generator.')

        count = 0
        h = inputs

        init_ops = []
        push_ops = []
        for b in range(self.model.num_blocks):
            for i in range(self.model.num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = 1
                else:
                    state_size = self.model.num_hidden
                    
                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

                state_ = q.dequeue()
                push = q.enqueue([h])
                init_ops.append(init)
                push_ops.append(push)

                h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
                count += 1

        outputs = _output_linear(h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops
        
        # Initialize queues.
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples):
        predictions = []
        for step in range(num_samples):

            feed_dict = {self.inputs: input}
            output = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0] # ignore push ops
            value = np.argmax(output[0, :])

            input = np.array(self.bins[value])[None, None] # TODO
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                plt.plot(predictions_[0, :], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions_ = np.concatenate(predictions, axis=1)
        return predictions_
"""

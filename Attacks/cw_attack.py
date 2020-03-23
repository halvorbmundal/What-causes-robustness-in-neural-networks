#Interface to run CW/EAD attacks

import numpy as np
from Attacks.li_attack import CarliniLi
from Attacks.l2_attack import CarliniL2
from Attacks.l1_attack import EADL1
from datasets.setup_cifar import CIFAR
from datasets.setup_tinyimagenet import TinyImagenet
from tensorflow.contrib.keras.api.keras.models import load_model
from utils import generate_data
import time as timer
import random
from train_resnet import *

def loss(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

#Runs CW/EAD attack with specified norm
def cw_attack(file_name, norm, sess, num_image=10, data_set_class=MNIST()):
    np.random.seed(1215)
    tf.set_random_seed(1215)
    random.seed(1215)
    if norm == '1':
        attack = EADL1
        norm_fn = lambda x: np.sum(np.abs(x),axis=(1,2,3))
    elif norm == '2':
        attack = CarliniL2
        norm_fn = lambda x: np.sum(x**2,axis=(1,2,3))
    elif norm == 'i':
        attack = CarliniLi
        norm_fn = lambda x: np.max(np.abs(x),axis=(1,2,3))

    data = data_set_class

    model = load_model(file_name, custom_objects={'fn':loss,'tf':tf, 'ResidualStart' : ResidualStart, 'ResidualStart2' : ResidualStart2})
    inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=num_image, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.predict, start=0)
    model.predict = model
    model.num_labels = 10

    model.image_size = data.test_data.shape[1]
    model.num_channels = data.test_data.shape[3]
    model.num_labels = data.test_labels.shape[1]
        
    
    start_time = timer.time()
    attack = attack(sess, model, max_iterations=1000)
    perturbed_input = attack.attack(inputs, targets)
    UB = np.average(norm_fn(perturbed_input-inputs))
    return UB, (timer.time()-start_time)/len(inputs)
    

import os
from models.shufflenet import ShuffleNet
from models.alexnet import AlexNet
import tensorflow as tf
import argparse
import sys
import tensorflow.contrib.slim as slim
from tensorflow.python.autograph.core.converter import apply_
from email.policy import default
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tqdm import tqdm
from utils import misc
from tensorflow.python.client import device_lib
import pdb
import numpy as np
import cv2 as cv

def parseArguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--height',default=128, type=int)
    parser.add_argument('--width',default=128, type=int)
    parser.add_argument('--batch_size',default=64, type=int)
    parser.add_argument('--dropout',default=0.8,type=float)
    parser.add_argument('--weight_decay',default=5e4,type=float)
    parser.add_argument('--img_dir',default='/home/nemo/imagenet/imagenet_train_128/')
    parser.add_argument('--num_epochs',default=300, type=int)
    parser.add_argument('--save_every',default=1, type=int)
    parser.add_argument('--validate_every',default=1, type=int)
    parser.add_argument('--learning_rate_file', default='./data/learning_rate.txt')
    parser.add_argument('--val_img_dir', default='/home/nemo/imagenet/imagenet_val_128/')
    parser.add_argument('--val_label_file', default='/home/nemo/imagenet/ILSVRC2012_validation_ground_truth.txt')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/')
    parser.add_argument('--need_resize', default=False, type=bool)
    parser.add_argument('--preprocess_multi_thread_num', default=8, type=int)
    parser.add_argument('--gpu_num', default='0,1')
    return parser.parse_args(sys.argv[1:]) 

args=parseArguments()

def inference1(x_batch, is_training):
    #build the network
    encoder = ShuffleNet(x_batch, num_classes=2, pretrained_path="", train_flag=is_training, weight_decay=args.weight_decay)
    encoder.build()
    with tf.variable_scope('final', reuse=False):
        net = slim.flatten(encoder.stage4, scope='flatten')
        print(net,net.op.name, net.shape.as_list())
        net = slim.dropout(net, args.dropout, is_training=is_training, scope='Dropout')
        print(net,net.op.name, net.shape.as_list())
        #weights_initializer=slim.l2_regularizer(args.weight_decay)
        batch_norm_params = {
            'decay': 0.995,
            'epsilon': 0.001,
            'updates_collections': None,
            'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
        }
        net=slim.fully_connected(net, 1000, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='Bottleneck')#, reuse=False)
        print(net,net.op.name, net.shape.as_list())
    pre_class=tf.argmax(net, axis=1)    
    return pre_class, net

def inference(x_batch, is_training):
    alexnet=AlexNet(x_batch,keep_prob=0.8, num_classes=1000, skip_layer=[])
    alexnet.create()
    net=alexnet.fc8
    pre_class=tf.argmax(net, axis=1)
    return pre_class, net

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def average_gradients(tower_grads): 
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is not None: 
                expanded_g = tf.expand_dims(g, 0) 
                grads.append(expanded_g)
        if len(grads) ==0:
            continue
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
    
def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    gpus=get_available_gpus()
    
    #init input
#     with tf.variable_scope('input'):
#         x_pl=tf.placeholder(dtype=tf.uint8, shape=[args.batch_size, args.height, args.width, 3], name='x_pl')
#         y_label=tf.placeholder(dtype=tf.int32, shape=[args.batch_size], name='y_label')
    learning_rate_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')
    train_placeholder = tf.placeholder(dtype=tf.bool, name='train_phase')
    test_input_placeholder = tf.placeholder(dtype=tf.uint8, name="test_input", shape=[None,args.height, args.width, 3])
    dataset, batch_num_one_epoch=get_dataset(args)
    iterator=dataset.make_initializable_iterator()
    x_batch, y_batch, image_path_batch= iterator.get_next()
    x_batch.set_shape((args.batch_size, args.height, args.width, 3))
    y_batch.set_shape((args.batch_size,))
    image_path_batch.set_shape((args.batch_size,))

    #start the train
    global_step=tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate_placeholder)
    with tf.variable_scope(tf.get_variable_scope()):
        pred_class,  net= inference(x_batch, train_placeholder)
        tf.get_variable_scope().reuse_variables()
        pred_class_test, _ = inference(test_input_placeholder, train_placeholder)
    match_result=tf.equal(pred_class, tf.cast(y_batch,tf.int64))
    match_sum=tf.reduce_sum(tf.cast(match_result,tf.float32))
    acc=match_sum/args.batch_size
    #build the loss
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch,logits=net)
    cross_entropy_loss=tf.reduce_mean(ce)
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss=cross_entropy_loss+regularization_loss
    train_op=optimizer.minimize(loss, global_step)
    
    saver = tf.train.Saver(max_to_keep=1)
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        save_path=tf.train.latest_checkpoint(args.checkpoint_dir)
        if save_path != None:
            print('restore from {}'.format(save_path))
            saver.restore(sess, save_path)
        accuracys=[]
        for i in range(args.num_epochs):
            sess.run(iterator.initializer)
            learning_rate=misc.get_learning_rate(args.learning_rate_file, i)
            feed_dict={learning_rate_placeholder:learning_rate, train_placeholder:True}
            print("lr={}".format(learning_rate))
            total_loss=[]
            total_ce_loss=[]
            total_re_loss=[]
            for i in tqdm(range(batch_num_one_epoch),desc="epoch-"+str(i)):
                loss_result, _,accuracy,ce_loss, re_loss = sess.run([loss,train_op,acc,cross_entropy_loss,regularization_loss], feed_dict=feed_dict)
                total_loss.append(loss_result)
                accuracys.append(accuracy)
                total_ce_loss.append(ce_loss)
                total_re_loss.append(re_loss)
            print("loss={}, acc={},ce_loss={},re_loss={}".format(np.mean(total_loss), np.mean(accuracys), np.mean(total_ce_loss),np.mean(total_re_loss)))
            if i % args.validate_every == 0:
                validate(sess,train_placeholder, test_input_placeholder, pred_class_test)
            if i % args.save_every == 0:
                saver.save(sess, args.checkpoint_dir, global_step)
                
def validate(sess,train_placeholder, test_input_placeholder, pred_class):
#     labels, paths = get_val_data(args)
    labels, paths = get_val_data_from_train_data()
    labels=labels[:500]
    paths=paths[:500]
    batch_num=len(paths)//args.batch_size
    pres=[]
    for i in tqdm(range(batch_num)):
        x_batch=[]
        path_batch=paths[i*args.batch_size:(i+1)*args.batch_size]
        for path in path_batch:
            img=cv.imread(path)
            img=img[...,::-1]
            x_batch.append(img)
        pre = sess.run(pred_class, feed_dict={train_placeholder:False, test_input_placeholder:x_batch})
        pres.extend(pre)
    tp=np.sum(np.where(np.array(labels[:len(pres)])-np.array(pres)==0,1,0))
    acc=tp/(batch_num*args.batch_size)
    print('acc={}'.format(acc))

def parse_dataset(image_path, label):
    image_string=tf.read_file(image_path)
    if tf.image.is_jpeg(image_string) == True:
        image=tf.image.decode_jpeg(image_string,channels=3)
    else:
        image=tf.image.decode_png(image_string,channels=3)
    if args.need_resize == True:
        image=tf.image.resize_images(image, (args.height,args.width))
    image = tf.cast(image,tf.uint8)
    label = tf.cast(label, tf.int32)
    return image, label, image_path

def get_dataset(args):
    img_paths=[]
    labels=[]
    class_names=os.listdir(args.img_dir)
    class_names.sort()
    for i, class_name in enumerate(class_names):
        file_names=os.listdir(os.path.join(args.img_dir,class_name))
        file_names.sort()
        img_paths.extend([os.path.join(args.img_dir,class_name,file_name) for file_name in file_names])
        labels.extend([i]*len(file_names))
    dataset=tf.data.Dataset.from_tensor_slices((img_paths,labels)).shuffle(buffer_size=len(img_paths))
    dataset=dataset.map(parse_dataset,num_parallel_calls=args.preprocess_multi_thread_num)
    batch_size=args.batch_size
    dataset=dataset.batch(batch_size)
    print("Dataset Size={}".format(len(img_paths)))
    return dataset, len(img_paths)//batch_size

def get_val_data_from_train_data():
    img_paths=[]
    labels=[]
    class_names=os.listdir(args.img_dir)
    class_names.sort()
    for i, class_name in enumerate(class_names):
        file_names=os.listdir(os.path.join(args.img_dir,class_name))
        img_paths.extend([os.path.join(args.img_dir,class_name,file_name) for file_name in file_names])
        labels.extend([i]*len(file_names))
    return labels, img_paths

def get_val_data(args):
    with open(args.val_label_file) as f:
        content=f.read()
        labels=content.split('\n')[:-1]
        labels=[int(str) for str in labels]
    names=os.listdir(args.val_img_dir)
    names.sort()
    paths=[os.path.join(args.val_img_dir,name) for name in names]
    return labels, paths
    
    

def get_train_op(loss, global_step, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
#     compute_grads = optimizer.compute_gradients(loss, tf.global_variables())
#     apply_grads = optimizer.apply_gradients(compute_grads, global_step)
#     with tf.control_dependencies():
    train_op=optimizer.minimize(loss, global_step)
    return train_op


main()

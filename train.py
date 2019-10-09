import os
from models.shufflenet import ShuffleNet
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
import imageio
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
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--need_resize', default=False, type=bool)
    parser.add_argument('--preprocess_multi_thread_num', default=8, type=int)
    parser.add_argument('--gpu_num', default='0,1')
    return parser.parse_args(sys.argv[1:]) 

args=parseArguments()

def inference(x_batch, is_training):
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
    x_batch, y_batch= iterator.get_next()
    x_batch.set_shape((args.batch_size, args.height, args.width, 3))
    y_batch.set_shape((args.batch_size,))
    x_batches=tf.split(x_batch, len(gpus))
    y_batches=tf.split(y_batch, len(gpus))

    #start the train
    global_step=tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate_placeholder)
    tower_losses=[]
    tower_grads=[]
    with tf.variable_scope(tf.get_variable_scope()):
        for i, device in enumerate(gpus):
            with tf.device(device):
                with tf.name_scope('tower_{}'.format(i)) as scope:
                    with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
                        x_batch_i=x_batches[i]
                        y_batch_i=y_batches[i]
                        _,  net= inference(x_batch_i, train_placeholder)
                        #build the loss
                        ce=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch_i,logits=net)
                        cross_entropy_loss=tf.reduce_mean(ce)
                        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        loss_i=cross_entropy_loss+regularization_loss
                        tower_losses.append(loss_i)
                        tf.get_variable_scope().reuse_variables()
                        grad_i=optimizer.compute_gradients(loss_i)
                        tower_grads.append(grad_i)
                    if i==0:
                        pred_class, _ = inference(test_input_placeholder, train_placeholder)
    loss=tf.reduce_mean(tower_losses)
    grads=average_gradients(tower_grads)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.apply_gradients(grads, global_step)
        
    saver = tf.train.Saver(max_to_keep=1)
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        save_path=tf.train.latest_checkpoint(args.checkpoint_dir)
        if save_path != None:
            print('restore from {}'.format(save_path))
            saver.restore(sess, save_path)
        for i in range(args.num_epochs):
            sess.run(iterator.initializer)
            learning_rate=misc.get_learning_rate(args.learning_rate_file, i)
            feed_dict={learning_rate_placeholder:learning_rate, train_placeholder:True}
#             for i in tqdm(range(batch_num_one_epoch)):
#                 sess.run(train_op, feed_dict=feed_dict)
            if i % args.validate_every == 0:
                validate(sess,train_placeholder, test_input_placeholder, pred_class)
            if i % args.save_every == 0:
                saver.save(sess, args.checkpoint_dir, global_step)
                
def validate(sess,train_placeholder, test_input_placeholder, pred_class):
    labels, paths = get_val_data(args)
    batch_num=len(paths)//args.batch_size
    pres=[]
    x_batch=[]
    for i in tqdm(range(batch_num)):
        path_batch=paths[i*args.batch_size:(i+1)*args.batch_size]
        for path in path_batch:
            img=cv.imread(path)
            img=img[...,::-1]
            x_batch.append(img)
        pre = sess.run(pred_class, feed_dict={train_placeholder:False, test_input_placeholder:x_batch})
        pres.extend(pre)
    tp=np.sum(np.where(lables-pres==0,1,0))
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
    return image, label

def get_dataset(args):
    img_paths=[]
    labels=[]
    class_names=os.listdir(args.img_dir)
    class_names.sort()
    for i, class_name in enumerate(class_names):
        file_names=os.listdir(os.path.join(args.img_dir,class_name))
        img_paths.extend([os.path.join(args.img_dir,class_name,file_name) for file_name in file_names])
        labels.extend([i]*len(file_names))
    dataset=tf.data.Dataset.from_tensor_slices((img_paths,labels)).shuffle(buffer_size=len(img_paths))
    dataset=dataset.map(parse_dataset,num_parallel_calls=args.preprocess_multi_thread_num)
    gpus=get_available_gpus()
    batch_size=len(gpus)*args.batch_size
    dataset=dataset.batch(batch_size)
    return dataset, len(img_paths)//batch_size

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

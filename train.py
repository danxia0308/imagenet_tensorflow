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
import pdb

def parseArguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--height',default=128)
    parser.add_argument('--width',default=128)
    parser.add_argument('--batch_size',default=64)
    parser.add_argument('--dropout',default=0.8)
    parser.add_argument('--weight_decay',default=5e4)
    parser.add_argument('--img_dir',default='/data/sophie_bak/imagenet_train/')
    parser.add_argument('--num_epochs',default=300)
    parser.add_argument('--save_every',default=1)
    parser.add_argument('--validate_every',default=1)
    parser.add_argument('--learning_rate_file', default='./data/learning_rate.txt')
    parser.add_argument('--val_img_dir', default='/data/sophie_bak/imagenet_val/')
    parser.add_argument('--val_label_file', default='/data/sophie_bak/ILSVRC2012_validation_ground_truth.txt')
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    return parser.parse_args(sys.argv[1:]) 

args=parseArguments()

def inference(x_batch, y_batch, is_training):
    #build the network
    encoder = ShuffleNet(x_batch, num_classes=2, pretrained_path="", train_flag=is_training, weight_decay=args.weight_decay)
    encoder.build()
    net = slim.flatten(encoder.stage4)
    net = slim.dropout(net, args.dropout, is_training=is_training)
    #weights_initializer=slim.l2_regularizer(args.weight_decay)
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    net=slim.fully_connected(net, 1000, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, 
                         trainable=is_training)
    #build the loss
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch,logits=net)
    pre_class=tf.argmax(ce, axis=1)
    cross_entropy_loss=tf.reduce_mean(ce)
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss=cross_entropy_loss+regularization_loss
    
    return pre_class, loss
    
def main():
    is_training=True
    
    
    #init input
#     with tf.variable_scope('input'):
#         x_pl=tf.placeholder(dtype=tf.uint8, shape=[args.batch_size, args.height, args.width, 3], name='x_pl')
#         y_label=tf.placeholder(dtype=tf.int32, shape=[args.batch_size], name='y_label')
    learning_rate_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')
    dataset, batch_num_one_epoch=get_dataset(args)
    iterator=dataset.make_initializable_iterator()
    x_batch, y_batch= iterator.get_next()
    x_batch.set_shape((args.batch_size, args.height, args.width, 3))
    y_batch.set_shape((args.batch_size,))
    
    pre_class, loss = inference(x_batch, y_batch, is_training)
    saver = tf.train.Saver(max_to_keep=1)
    #start the train
    global_step=tf.train.get_or_create_global_step()
    train_op=get_train_op(loss, global_step, learning_rate_placeholder)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        save_path=tf.train.latest_checkpoint(args.checkpoint_dir)
        if save_path != None:
            print('restore from {}'.format(save_path))
            saver.restore(sess, save_path)
        for i in range(args.num_epochs):
            sess.run(iterator.initializer)
            learning_rate=misc.get_learning_rate(args.learning_rate_file, i)
            
            for i in tqdm(range(batch_num_one_epoch)):
                sess.run(train_op, feed_dict={learning_rate_placeholder:learning_rate})
            if i % args.validate_every == 0:
                validate(sess)
            if i % args.save_every == 0:
                saver.save(sess, args.checkpoint_dir, global_step)
                
def validate(sess):
    dataset, batch_num = get_val_dataset(args)
    labels=[]
    pres=[]
    x_batch, y_batch = iterator.get_next()
    pre_class, _ = inference(x_batch, y_batch, False)
    for i in tqdm(range(batch_num)):
        pre = sess.run(pre_class)
        pres.extend(pre)
        labels.extend(y_batch)
    tp=np.sum(np.where(lables-pres==0,1,0))
    acc=tp/(batch_num*args.batch_size)
    print('acc={}'.format(acc))

def parse_dataset(image_path, label):
    image_string=tf.read_file(image_path)
    if tf.image.is_jpeg(image_string) == True:
        image=tf.image.decode_jpeg(image_string)
    else:
        image=tf.image.decode_png(image_string)
    image_resized=tf.image.resize_images(image, (args.height,args.width))
    image_resized = tf.cast(image_resized,tf.uint8)
    label = tf.cast(label, tf.int32)
    return image_resized, label

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
    dataset=dataset.map(parse_dataset)
    dataset=dataset.batch(args.batch_size)
    return dataset, len(img_paths)//args.batch_size

def get_val_dataset(args):
    with open(args.val_label_file) as f:
        content=f.read()
        labels=content.split('\n')
    names=os.listdir(args.val_img_dir)
    names.sort()
    dataset = tf.data.Dataset.from_tensor_slices((names,labels))
    dataset = dataset.map(parse_dataset).batch(args.batch_size)
    return dataset, len(img_paths)//args.batch_size
    
    

def get_train_op(loss, global_step, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
#     compute_grads = optimizer.compute_gradients(loss, tf.global_variables())
#     apply_grads = optimizer.apply_gradients(compute_grads, global_step)
#     with tf.control_dependencies():
    train_op=optimizer.minimize(loss, global_step)
    return train_op


main()

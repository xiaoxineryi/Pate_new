import gzip
import os
import sys
from six.moves import urllib
import numpy as np
import tensorflow.compat.v1 as tf


FLAGS = tf.flags.FLAGS

def create_dir_if_needed(dest_directory):
  """Create directory if doesn't exist."""
  if not tf.gfile.IsDirectory(dest_directory):
    tf.gfile.MakeDirs(dest_directory)

  return True


def extract_mnist_data(filename, num_images, image_size, pixel_depth):
    """
    会将对应的数据变成一个四维数组：[image_index,y,x,channels]
    """
    if not tf.gfile.Exists(filename+'.npy'):
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            # 读取为buf
            buf = bytestream.read(image_size * image_size * num_images)
            # 将buf转换为np数组
            data = np.frombuffer(buf,dtype = np.uint8).astype(np.float32)
            data = (data-(pixel_depth/2.0))
            # 将数据转换为我们所需要的格式
            data = data.reshape(num_images,image_size,image_size,1)
            np.save(filename,data)
            return data
    else:
        with tf.gfile.Open(filename+'.npy',mode="rb") as file_obj:
            return np.load(file_obj)

def extract_mnist_labels(filename, num_images):
  """
  Extract the labels into a vector of int64 label IDs.
  """
  if not tf.gfile.Exists(filename+'.npy'):
    with gzip.open(filename) as bytestream:
      bytestream.read(8)
      buf = bytestream.read(1 * num_images)
      labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
      np.save(filename, labels)
    return labels
  else:
    with tf.gfile.Open(filename+'.npy', mode='rb') as file_obj:
      return np.load(file_obj)

def load_mnist(test_only = False):
    # 文件地址
    file_urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
                 ]
    # 下载并且获得对应的文件位置
    local_urls = maybe_download(file_urls, FLAGS.data_dir)

    # 解压对应的文件
    train_data = extract_mnist_data(local_urls[0], 60000, 28, 1)
    train_labels = extract_mnist_labels(local_urls[1], 60000)
    test_data = extract_mnist_data(local_urls[2], 10000, 28, 1)
    test_labels = extract_mnist_labels(local_urls[3], 10000)

    if test_only:
        return test_data, test_labels
    else:
        return train_data, train_labels, test_data, test_labels

def maybe_download(file_urls,directory):
    """
    如果当前需要下载的文件不存在的话，就下载对应的文件
    :param file_urls: 要下载文件的位置
    :param directory: 要存放的位置
    :return:
    """

    # 如果文件夹不存在就创建对应文件夹
    assert create_dir_if_needed(directory)

    #
    result = []

    for file_url in file_urls:
        # 查找对应的文件名
        fileName = file_url.split('/')[-1]

        # 如果是从Github上下载，那么就从本地名中去除对应?raw=true
        if fileName.endswith("?raw=true"):
            fileName = fileName[:-9]
        filePath = directory + '/' + fileName

        result.append(filePath)

        # 如果文件不存在的话，那么就下载
        if not tf.gfile.Exists(filePath):
            def _progress(count,block_size,total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (fileName,float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filePath,_ = urllib.request.urlretrieve(file_url,filePath,_progress)
            print()
            statinfo = os.stat(filePath)
            print('Successfully downloaded', fileName, statinfo.st_size, 'bytes.')
    return result

def partition_dataset(data,labels,nb_teachers,teacher_id):
    """
    给不同的教师分配不同的数据
    """
    assert len(data) == len(labels)
    assert int(teacher_id) < int(nb_teachers)

    # 获得一批次对应的数目
    batch_len = int(len(data) / nb_teachers)

    # 计算起止位置
    start = teacher_id * batch_len
    end = (teacher_id+1) * batch_len

    # 进行划分
    partition_data = data[start:end]
    partition_labels = labels[start:end]

    return partition_data, partition_labels
import tensorflow.compat.v1 as tf
import Input
import deep_cnn
import analysis
## 设置一些常量

tf.flags.DEFINE_string('dataset','mnist','数据集名称')
tf.flags.DEFINE_integer('nb_labels',10,'标签种类个数')

tf.flags.DEFINE_string('data_dir','./tmp','存放样本地址')
tf.flags.DEFINE_string('train_dir','./tmp/train_dir',
                       "存放训练中间数据地址")

tf.flags.DEFINE_integer('max_steps',3000,"最大训练次数")
tf.flags.DEFINE_integer('nb_teachers',50,'教师模型数目')
tf.flags.DEFINE_integer('teacher_id',0,'教师模型ID')



FLAGS = tf.flags.FLAGS


def train_teacher(dataset,nb_teachers,teacher_id):
    """
    训练指定ID的教师模型
    :param dataset: 数据集名称
    :param nb_teachers: 老师数量
    :param teacher_id: 老师ID
    :return:
    """
    # 如果目录不存在就创建对应的目录
    assert Input.create_dir_if_needed(FLAGS.data_dir)
    assert Input.create_dir_if_needed(FLAGS.train_dir)
    # 加载对应的数据集
    if dataset == 'mnist':
        train_data,train_labels,test_data,test_labels = Input.load_mnist()
    else:
        print("没有对应的数据集")
        return False

    # 给对应的老师分配对应的数据
    data, labels = Input.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)
    print("Length of training data: " + str(len(labels)))

    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt'
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + filename

    # 开始训练,并保存训练模型
    assert deep_cnn.train(data, labels, ckpt_path)

    # 拼接得到训练后的模型位置
    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

    # 读取教师模型对测试数据进行验证
    teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)
    # 计算教师模型准确率
    precision = analysis.accuracy(teacher_preds, test_labels)
    print('Precision of teacher after training: ' + str(precision))


    return True

def predict(dataset,nb_teachers,teacher_id):
    if dataset == 'mnist':
        train_data,train_labels,test_data,test_labels = Input.load_mnist()
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt'
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + filename

    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)
    # 读取教师模型对测试数据进行验证
    teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)
    precision = analysis.accuracy(teacher_preds, test_labels)
    print('Precision of teacher after training: ' + str(precision))

# 执行主方法
def main(argv = None):

    # assert train_teacher(FLAGS.dataset,FLAGS.nb_teachers,FLAGS.teacher_id)
    train_teacher(FLAGS.dataset,FLAGS.nb_teachers,FLAGS.teacher_id)
# 方法入口
if __name__ == '__main__':
    tf.app.run()
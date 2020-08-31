import numpy as np
import tensorflow.compat.v1 as tf
FLAGS = tf.flags.FLAGS
import Input
import deep_cnn
import analysis
import Aggregation

tf.flags.DEFINE_string('dataset','mnist',"数据集")
tf.flags.DEFINE_integer('nb_labels',10,"标签种类个数")

tf.flags.DEFINE_string('data_dir','./tmp',"数据集所在位置")
tf.flags.DEFINE_string('train_dir','./tmp/student_train_dir',"学生模型保存的位置")
tf.flags.DEFINE_string('teachers_dir','./tmp/train_dir',"教师模型保存的位置")

tf.flags.DEFINE_integer('teachers_max_steps',3000,'教师模型最大训练次数')

tf.flags.DEFINE_integer('max_steps',3000,"学生模型最大训练次数")
tf.flags.DEFINE_integer("nb_teachers",100,"教师模型个数")
tf.flags.DEFINE_integer("stdnt_share",5000,"学生模型所需要的数据个数")
tf.flags.DEFINE_integer("lap_scale",10,"拉普拉斯噪音维度")
tf.flags.DEFINE_boolean("save_labels",False,"是否保存教师模型投票结果")


def ensemble_preds(dataset, nb_teachers, stdnt_data):
    # 得到的数据规模是：教师模型个数、学生未标记数据个数，标签类别。
    # 最后得到的应该是每一个数据对应每一种标签的概率
    result_shape = (nb_teachers, len(stdnt_data), FLAGS.nb_labels)
    result = np.zeros(result_shape, dtype=np.float32)

    for teacher_id in range(nb_teachers):
        # 得到对应的教师模型位置
        ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(
            teacher_id) + '.ckpt-' + str(FLAGS.teachers_max_steps - 1)
        result[teacher_id] =deep_cnn.softmax_preds(stdnt_data,ckpt_path,return_logits=False)
        print("Computed Teacher " + str(teacher_id) + " softmax predictions")
    # print(result[2][0])
    return result

def prepare_student_data(dataset, nb_teachers, save):
    """
    准备学生模型数据，在这里进行聚合的修改
    """
    assert Input.create_dir_if_needed(FLAGS.train_dir)

    if dataset == 'mnist':
        test_data,test_labels = Input.load_mnist(test_only=True)
    else:
        return False

    assert FLAGS.stdnt_share < len(test_data)
    stdnt_data = test_data[:FLAGS.stdnt_share]
    # 得到的数据是 教师id,无标签数据个数，以及每个标签的概率
    teacher_preds = ensemble_preds(dataset,nb_teachers,stdnt_data)
    # 得到教师模型聚合后的结果 不可信的数据标签为-1
    student_labels = Aggregation.noisy_max_plus(teacher_preds,FLAGS.lap_scale,reliability=0.1,gap=10)
    ans_labels = test_labels[:FLAGS.stdnt_share]
    indexs = [i for i in range(len(student_labels)) if student_labels[i] == -1]
    print("the -1 indexs are")
    print(indexs)
    # 删除对应元素
    student_data = test_data[:FLAGS.stdnt_share]
    student_data = np.delete(student_data,indexs,axis=0)
    print("len of student_data is"+str(len(student_data)))
    ans_labels = np.delete(ans_labels,indexs)
    student_labels = np.delete(student_labels,indexs)
    print("len of student_labels is"+str(len(student_labels)))

    ac_ag_labels = analysis.accuracy(student_labels,ans_labels)
    print("Accuracy of the aggregated labels: " + str(ac_ag_labels))

    stdnt_test_data = test_data[FLAGS.stdnt_share:]
    stdnt_test_labels = test_labels[FLAGS.stdnt_share:]

    return student_data, student_labels, stdnt_test_data, stdnt_test_labels

def train_student(dataset, nb_teachers):

    assert Input.create_dir_if_needed(FLAGS.train_dir)

    # 准备学生模型数据
    student_dataset = prepare_student_data(dataset,nb_teachers,save=True)
    # 解压学生数据
    stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = student_dataset

    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student.ckpt'
    # 训练
    assert deep_cnn.train(stdnt_data, stdnt_labels, ckpt_path)

    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

    # 预测
    student_preds = deep_cnn.softmax_preds(stdnt_test_data,ckpt_path_final)

    precision = analysis.accuracy(student_preds,stdnt_test_labels)
    print('Precision of student after training: ' + str(precision))

    return True


def main(argv=None): # pylint: disable=unused-argument
  # Run student training according to values specified in flags
  assert train_student(FLAGS.dataset, FLAGS.nb_teachers)

if __name__ == '__main__':
    tf.app.run()
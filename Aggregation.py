import math
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

def noisy_max_plus(teacher_preds, lap_scale,reliability):
    untrusted_time = np.zeros(FLAGS.stdnt_share)
    # votes表示全部教师投票的结果，其为二维数组是 数据ID，数据标签投票数
    votes = np.zeros((FLAGS.stdnt_share,FLAGS.nb_labels))
    # 如果最大的两个预测结果概率相差大于0.1，那么就说明可信，直接添加。
    for teacher_id in range(FLAGS.nb_teachers):
        for index,predict in enumerate(teacher_preds[teacher_id]):
            b = sorted(enumerate(predict),key=lambda x:x[1])
            if b[-1][1] - b[-2][1] > reliability:
                votes[index][b[-1][0]] += 1
            else:
                untrusted_time[index] += 1
    # 记录结果
    results = np.zeros(FLAGS.stdnt_share,dtype=np.int32)
    for i,vote in enumerate(votes) :
        for index in range(FLAGS.nb_labels):
            vote[index] += np.random.laplace(loc=0.0,scale=float(lap_scale))
        results[i] = np.argmax(vote)
    print("the untrusted time is "+str(untrusted_time))
    print(results)
    return results

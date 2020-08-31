import os
for i in range(100):
    string = "python train_teachers.py --nb_teachers=100 --teacher_id=" + str(i) +" --dataset=mnist"
    os.system(string)

    print(string)
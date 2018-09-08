import os
os.system("python train_test_combine/train_test_combine_main.py")
os.system("python uid_pos_neg_feature/uid_pos_neg_main.py")
os.system("python cv_feature/cv_main.py")
os.system("python transfer_feature/transfer_main.py")
os.system("python cv_NN/countvector_NN.py")
os.system("python LGB/lgb_transfer_feature.py")

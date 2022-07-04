import argparse
import os
import tensorflow as tf
from datasets import Dataset
from models import Mask_G, Face_G, Face_D
from train import Train_Mask, Train_Face
from test import Test

import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Train the Mask removal network',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', default='multi_test', choices=['mask-train', 'face-train', 'single-test', 'multi-test'], dest='mode')
    parser.add_argument('--dir_test', default='./test', dest='dir_test')
    parser.add_argument('--ckpt_num', default=None, dest='ckpt_num')
    parser.add_argument('--choice_ckpt', dest='choice_ckpt',  action='store_true')
    parser.add_argument('--no-choice_ckpt', dest='choice_ckpt', action='store_false')
    parser.set_defaults(choice_ckpt=False)
    
    PATH = os.getcwd()
    train_path = PATH+'/train'                
    test_path = parser.parse_args().dir_test
    mask_checkpoint_dir = PATH+'/mask_checkpoints'
    face_checkpoint_dir = PATH+'/face_checkpoints'
    
    mask_G, face_G, face_D = Mask_G(filters=32), Face_G(filters=32), Face_D(filters=32)
    
    if parser.parse_args().mode == 'single-test':     # Visualize one image in a folder
        test = Test(mask_G, face_G, face_D, img_dir=test_path, mask_checkpoint_dir=mask_checkpoint_dir, face_checkpoint_dir=face_checkpoint_dir)
        test.one_predict()
    elif parser.parse_args().mode == 'multi-test':    # Visualize up to 4 images in a folder
        test = Test(mask_G, face_G, face_D, img_dir=test_path, mask_checkpoint_dir=mask_checkpoint_dir, face_checkpoint_dir=face_checkpoint_dir)
        test.multiple_predict()
    else:
        BATCH_SIZE = 4
        trainset = Dataset(file_path=train_path, batch_size=BATCH_SIZE)
        trainset = trainset.make_train()
        with tf.device('/gpu:0'):
            if parser.parse_args().mode == 'mask-train':
                mask_train = Train_Mask(mask_G, checkpoint_dir=mask_checkpoint_dir)
                if parser.parse_args().choice_ckpt :
                    mask_train.load(checkpoint_dir=mask_checkpoint_dir, ckpt_num=parser.parse_args().ckpt_num)
                mask_train.fit(trainset, epochs=5)
                
            elif parser.parse_args().mode == 'face-train':       
                face_train = Train_Face(mask_G, face_G, face_D, 
                                mask_checkpoint_dir=mask_checkpoint_dir, face_checkpoint_dir=face_checkpoint_dir)
                if parser.parse_args().choice_ckpt :
                    face_train.load(checkpoint_dir=face_checkpoint_dir, ckpt_num=parser.parse_args().ckpt_num)
                face_train.fit(trainset, epochs=5)

if __name__ == '__main__':
    main()
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import noise_processing
from datasets import Dataset

class Test:
    def __init__(self, mask_model, face_model, dis_model, img_dir=None, mask_checkpoint_dir=None, face_checkpoint_dir=None):
        plt.close()
        plt.ioff()
        self.mask_model = mask_model
        self.face_model = face_model
        self.dis_model = dis_model
        self.mask_model.build(input_shape=(None, 128, 128, 3))
        self.face_model.build(input_shape=(None, 128, 128, 3))
        self.dis_model.build(input_shape=[(None, 128, 128, 3), (None, 128, 128, 3)])
        self.optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.img_dir = img_dir
        self.mask_checkpoint_dir = mask_checkpoint_dir
        self.face_checkpoint_dir = face_checkpoint_dir
        
        self.mask_checkpoint = tf.train.Checkpoint(generator_optimizer=self.optimizer,
                                        generator=self.mask_model)
        self.mask_checkpoint.restore(tf.train.latest_checkpoint(self.mask_checkpoint_dir))
        
        self.face_checkpoint = tf.train.Checkpoint(generator_optimizer=self.optimizer,
                                            discriminator_optimizer=self.optimizer, 
                                            generator=self.face_model,
                                            discriminator=self.dis_model)
        self.face_checkpoint.restore(tf.train.latest_checkpoint(self.face_checkpoint_dir))

    def one_predict(self):        
        img = plt.imread(self.img_dir)
        if self.img_dir.endswith('.png'):
            img = img * 255.0
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, [128, 128],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.reshape(img, [1, 128, 128, 3])
        img = tf.cast(img, tf.float32)
        img = (img / 127.5) - 1
        
        mask = self.mask_model(img, training=False)
        process_img = noise_processing(img, mask)
        pred = self.face_model(process_img, training=False)

        plt.figure(figsize=(8,8))
        plt.subplot(1, 2, 1)
        plt.title('Image with mask')
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Prediction Image')
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred[0]))
        plt.axis('off')
        plt.show()
        
    def multiple_predict(self):     
        testset = Dataset(file_path=self.img_dir, batch_size=1)
        testset = testset.make_test()
        img_num = len(testset)
        
        plt.figure(figsize=(10, 10))
        plt.suptitle('Prediction Image', fontsize=20, y=0.7)
        
        for i, img in enumerate(testset):
            mask = self.mask_model(img, training=False)
            process_img = noise_processing(img, mask)
            pred = self.face_model(process_img, training=False)
            pred = tf.concat([img, pred], axis=1)

            plt.subplot(1, img_num, i+1)
            plt.imshow(pred[0] * 0.5 + 0.5)
            plt.axis('off') 
        plt.show()
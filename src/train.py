import time
import os
import tensorflow as tf
from tqdm import tqdm
from datasets import Dataset
from models import Mask_G, Face_G, Face_D
from utils import *

class Train_Mask:
    def __init__(self, model, alpha=100, lr=2e-4, checkpoint_dir=None):
        self.model = model
        self.model.build(input_shape=(None, 128, 128, 3))
        self.alpha = alpha
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        self.gen_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.optimizer,
                                        generator=self.model)
        
    def generator_loss(self, gen_output, target):
        # Binary cross entropy
        gan_loss = self.gen_loss(target, gen_output)
        # Mean absolute error
        l1_loss = self.alpha * tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.alpha * l1_loss)

        return total_gen_loss

    # @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape:
            gen_output = self.model(input_image, training=True)
            gen_loss = self.generator_loss(gen_output, target)
        generator_gradients = gen_tape.gradient(gen_loss,
                                                self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(generator_gradients,
                                            self.model.trainable_variables))
        
        return gen_loss.numpy()
    
    def save(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.checkpoint.save(file_prefix= self.checkpoint_prefix)

    def load(self, checkpoint_dir, ckpt_num=None):
        if ckpt_num is None:
            self.face_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        else:
            self.face_checkpoint.restore(checkpoint_dir+'/ckpt-{ckpt_num}'.format(ckpt_num=ckpt_num))
        
    def fit(self, dataset, epochs):     
        for epoch in range(epochs):
            pbar = tqdm(enumerate(dataset), total=len(dataset), desc='epoch', ncols=80)   
            pbar.set_description(f'{epoch+1} epoch')
            for step, (_, mask_input ,binary_input) in pbar:
                loss = self.train_step(mask_input, binary_input)
                # and pbar loss update
                if step % 20 == 0:
                    pbar.set_postfix(loss=loss)    
                # training_visualization once per 500 step
                if step % 500 == 0:
                    training_visualization(self.model, mask_input, binary_input, epoch, step, 'mask')     
            # Save (checkpoint) the model once per 2 epcoh
            if (epoch + 1) % 1 == 0:
                self.save(self.checkpoint_dir)

class Train_Face:
    def __init__(self, mask_model, face_model, dis_model, alpha=100, lr=2e-4, mask_checkpoint_dir=None, face_checkpoint_dir=None):
        self.mask_model = mask_model
        self.face_model = face_model
        self.dis_model = dis_model
        self.mask_model.build(input_shape=(None, 128, 128, 3))
        self.face_model.build(input_shape=(None, 128, 128, 3))
        self.dis_model.build(input_shape=[(None, 128, 128, 3), (None, 128, 128, 3)])

        self.alpha = alpha
        self.lr = lr
        self.gan_BCE_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mask_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
        self.gen_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)
        self.dis_optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.5)

        self.mask_checkpoint_dir = mask_checkpoint_dir
        self.face_checkpoint_dir = face_checkpoint_dir
        self.face_checkpoint_prefix = os.path.join(self.face_checkpoint_dir, "ckpt")
        
        self.face_checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
                                                  discriminator_optimizer=self.dis_optimizer, 
                                                  generator=self.face_model,
                                                  discriminator=self.dis_model)
        
        self.mask_checkpoint = tf.train.Checkpoint(generator_optimizer=self.mask_optimizer,
                                        generator=self.mask_model)
        self.mask_checkpoint.restore(tf.train.latest_checkpoint(self.mask_checkpoint_dir))

    def generator_loss(self, disc_generated_output, gen_output, target):
        # Binary cross entropy
        # gan_loss = self.gen_loss(tf.ones_like(disc_generated_output), disc_generated_output)
        # ssim loss
        ssim_loss = tf.image.ssim(target, gen_output, max_val=1.0)
        # Mean absolute error
        l1_loss = self.alpha * tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = ssim_loss + (self.alpha * l1_loss)
        
        return total_gen_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.gan_BCE_loss(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.gan_BCE_loss(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        
        return total_disc_loss  

    # @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.face_model(input_image, training=True)
            
            disc_real_output = self.dis_model([input_image, target], training=True)
            disc_generated_output = self.dis_model([input_image, gen_output], training=True)
        
            gen_total_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.face_model.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss,
                                                self.dis_model.trainable_variables)
            
            self.gen_optimizer.apply_gradients(zip(generator_gradients,
                                              self.face_model.trainable_variables))
            self.dis_optimizer.apply_gradients(zip(discriminator_gradients,
                                              self.dis_model.trainable_variables))
            
        return gen_total_loss.numpy(), disc_loss.numpy()

    def save(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.face_checkpoint.save(file_prefix= self.face_checkpoint_prefix)

    def load(self, checkpoint_dir, ckpt_num=None):
        if ckpt_num is None:
            self.face_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        else:
            self.face_checkpoint.restore(checkpoint_dir+'/ckpt-{ckpt_num}'.format(ckpt_num=ckpt_num))

    def fit(self, dataset, epochs):     
        for epoch in range(epochs):
            pbar = tqdm(enumerate(dataset), total=len(dataset), desc='epoch')   
            pbar.set_description(f'{epoch+1} epoch')
            for step, (real_input, mask_input ,_) in pbar:
                gen_output = self.mask_model(mask_input, training=False)
                process_img = noise_processing(mask_input, gen_output)
                gen_loss, dis_loss = self.train_step(process_img, real_input)
                # # and pbar loss update
                if step % 20 == 0:
                    pbar.set_postfix(gen_loss=gen_loss, dis_loss=dis_loss)    
                # training_visualization once per 500 step
                if step % 500 == 0:
                    training_visualization(self.face_model, process_img, real_input, epoch, step, 'face')     
            # Save (checkpoint) the model once per 2 epcoh
            if (epoch + 1) % 2 == 0:
                self.save(self.face_checkpoint_dir)
                
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.ion()
figure, ax = plt.subplots(figsize=(8,8))

np.random.seed(42)

def noise_processing(mask_image, generate_image):
    """
    Mask Generator를 통해 나온 Binary 이미지에 원본 이미지를 합성하여 Noise 생성
    Args:
      mask_image : 마스크 쓴 인물 이미지
      generate_image : model를 통해 생성된 이미지
    Return:
      Noiser가 된 인물 이미지
    """
    images = []
    height, width = 128, 128
    mask_image = mask_image.numpy()
    generate_image = generate_image.numpy()
    noise = np.random.rand(height, width, 3)*255.0
    batch = mask_image.shape[0]

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    for i in range(batch):
        image = mask_image[i, :, :, :]
        mask = generate_image[i, :, :, :]
        mask = cv2.erode(mask, k)             #  mask  Morphology 연산 전처리
        mask = cv2.dilate(mask, k)

        mask = (mask + 1) * 127.5
        image = (image + 1) * 127.5
        # image = image.numpy()
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)   #  mask 전처리
        
        # Masking된 RGB채널에 노이즈 생성
        image1 = np.where(mask[:,:,0] == 0, image[:, :, 0], noise[:,:,0])
        image2 = np.where(mask[:,:,1] == 0, image[:, :, 1], noise[:,:,1])
        image3 = np.where(mask[:,:,2] == 0, image[:, :, 2], noise[:,:,2])

        image1 = image1[:, :, np.newaxis]
        image2 = image2[:, :, np.newaxis]
        image3 = image3[:, :, np.newaxis]

        noise_image = np.concatenate([image1, image2], axis=-1)
        noise_image = np.concatenate([noise_image, image3], axis=-1)

        images.append(noise_image[np.newaxis, :, :, :])

    image_input = np.array(images).reshape((batch, height, width, 3))
    image_input = tf.convert_to_tensor(image_input, dtype=tf.float32)
    
    return (image_input / 127.5) - 1

def training_visualization(model, test_input, tar, epoch, step , types='mask'):
    """
    training visualization
    Args:
        model : Generate model
    """
    if types == 'mask':
      save_dir = './mask_training'
    elif types == 'face':
      save_dir = './face_training'
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)  
      
    prediction = model(test_input, training=False)
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    plt.savefig(save_dir+"/epoch_{epoch}_step_{step}.png".format(epoch=epoch+1, step=step+1))
    figure.canvas.draw()
    figure.canvas.flush_events()
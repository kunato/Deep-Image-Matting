import math
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from trimap import get_trimap
from model import build_encoder_decoder, build_refinement
from utils import compute_mse_loss, compute_sad_loss
from utils import get_final_output, safe_crop, draw_str


if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4

    pretrained_path = 'models/final.42-0.0398.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)
    print(final.summary())

    out_test_path = 'data/predict/'
    test_images = [f for f in os.listdir(out_test_path) if
                   os.path.isfile(os.path.join(out_test_path, f)) and f.endswith('.png')]
    samples = random.sample(test_images, 1)

    total_loss = 0.0
    for i in range(len(samples)):
        filename = samples[i]
        image_name = filename.split('.')[0]

        print('\nStart processing image: {}'.format(filename))

        bgr_img = cv.imread(os.path.join(out_test_path, filename))
        bg_h, bg_w = bgr_img.shape[:2]
        print('bg_h, bg_w: ' + str((bg_h, bg_w)))

        trimap = get_trimap(image_name)

        cv.imwrite('out/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
        cv.imwrite('out/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))

        x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
        x_test[0, :, :, 0:3] = bgr_img / 255.
        x_test[0, :, :, 3] = trimap / 255.

        y_pred = final.predict(x_test)
        
        y_pred = np.reshape(y_pred, (img_rows, img_cols))
        print(y_pred.shape)
        y_pred = y_pred * 255.0
        y_pred = get_final_output(y_pred, trimap)
        y_pred = y_pred.astype(np.uint8)

        out = y_pred.copy()
        cv.imwrite('out/{}_out.png'.format(i), out)

    K.clear_session()

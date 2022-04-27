"""All the usefull functions, currently in a single file"""

import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def compress_splits(trainx, trainy, testx, testy, dir='data/compressed/'):
    """Compresses train and test splits to .npz"""
    trainx_ = trainx.numpy()
    trainy_ = trainy.numpy()
    testx_ = testx.numpy()
    testy_ = testy.numpy()
    np.savez_compressed(dir + 'trainx.npz', trainx_)
    np.savez_compressed(dir + 'trainy.npz', trainy_)
    np.savez_compressed(dir + 'testx.npz', testx_)
    np.savez_compressed(dir + 'testy.npz', testy_)


def uncompress_splits(dir='data/compressed/'):
    """Uncompresses train and test splits from .npz"""
    trainx = np.load(dir + 'trainx.npz')
    trainy = np.load(dir + 'trainy.npz')
    testx = np.load(dir + 'testx.npz')
    testy = np.load(dir + 'testy.npz')

    trainx = trainx['arr_0']
    trainy = trainy['arr_0']
    testx = testx['arr_0']
    testy = testy['arr_0']

    trainx = tf.convert_to_tensor(trainx, dtype=tf.float32)
    trainy = tf.convert_to_tensor(trainy, dtype=tf.float32)
    testx = tf.convert_to_tensor(testx, dtype=tf.float32)
    testy = tf.convert_to_tensor(testy, dtype=tf.float32)

    return trainx, trainy, testx, testy


mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


def prep_image(path):
    """Giga chad func to detect face on image, annotate it 
    and return cropped image and annotations.
    Uses MediaPipe"""
    IMAGE_FILES = [path]
    try:
        with mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as face_detection:

            for _, file in enumerate(IMAGE_FILES):
                image = cv2.imread(file)
                results = face_detection.process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.detections:
                    continue

                for detection in results.detections:
                    location_data = detection.location_data

                    bb = location_data.relative_bounding_box
                    bb = [
                        int(bb.xmin * image.shape[0]
                            ), int(bb.ymin*image.shape[1]),
                        int((bb.width + bb.xmin) *
                            image.shape[0]), int((bb.height + bb.ymin) * image.shape[1]),
                    ]
                    image = image[bb[1]: bb[3], bb[0]:bb[2]]
                    break
                with mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5) as face_mesh:

                    results = face_mesh.process(image)

                    if not results.multi_face_landmarks:
                        continue

                    x = []
                    y = []
                    z = []
                    for face_landmarks in results.multi_face_landmarks:
                        for i in range(478):
                            x.append(face_landmarks.landmark[i].x)
                            y.append(face_landmarks.landmark[i].y)
                            z.append(face_landmarks.landmark[i].z)

        image = Image.fromarray(image)
        image = image.resize((128, 128))
        image = np.array(image)

        image = tf.convert_to_tensor(image, dtype=tf.float32) / 255

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        z = tf.convert_to_tensor(z, dtype=tf.float32)

        return image, tf.convert_to_tensor([x, y, z], dtype=tf.float32)
    except Exception as err:
        print(f'cought exc in{path}')
        return

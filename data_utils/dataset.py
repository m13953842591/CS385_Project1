import numpy as np
from config import *
import os
import cv2
from data_utils.crop import crop
from data_utils.hog import hog_skimage


def get_image_dataset():
    out_dir_train = os.path.join(DATA_PATH, 'image\\train')
    out_dir_test = os.path.join(DATA_PATH, 'image\\test')
    if not os.path.exists(out_dir_train):
        os.mkdir(out_dir_train)
    if not os.path.exists(out_dir_test):
        os.mkdir(out_dir_test)

    fddb_path = os.path.join(DATA_PATH, 'FDDB-folds')
    image_path = os.path.join(DATA_PATH, 'originalPics')

    for i in range(1, 11):
        filename = os.path.join(fddb_path, "FDDB-fold-%02d-ellipseList.txt" % i)
        print("reading FDDB-fold-%02d-ellipseList.txt ..." % i)
        with open(filename, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                img_name = "_".join(line[:-1].split('/'))
                line = line[:-1] + ".jpg"
                img = cv2.imread(os.path.join(image_path, line))
                num_face = int(f.readline()[:-1])
                if img is None:
                    print("error, image: %s not found" % os.path.join(
                            image_path, line))
                    for _ in range(num_face):
                        f.readline()
                    continue

                # image exist
                for _ in range(num_face):
                    annot = f.readline().split(' ')
                    face_img_pos = crop(img, float(annot[3]), float(annot[4]),
                                    float(annot[0]), float(annot[1]),
                                    positive=True)

                    out_dir = out_dir_train if i < 9 else out_dir_test
                    cv2.imwrite(
                        os.path.join(out_dir, img_name + ".00.jpg"),
                        face_img_pos)

                    if i < 5 or i == 10:
                        face_img_neg = crop(img,
                                            float(annot[3]),
                                            float(annot[4]),
                                            float(annot[0]),
                                            float(annot[1]),
                                            positive=False)
                        out_dir = out_dir_train if i < 5 else out_dir_test
                        for j in range(9):
                            cv2.imwrite(os.path.join(out_dir,
                                                     img_name + ".%02d.jpg" % (j + 1)),
                                        face_img_neg[j])


def get_hog_dataset():
    out_dir = os.path.join(DATA_PATH, 'hog')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    fddb_path = os.path.join(DATA_PATH, 'FDDB-folds')
    image_path = os.path.join(DATA_PATH, 'originalPics')

    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []

    for i in range(1, 11):
        filename = os.path.join(fddb_path, "FDDB-fold-%02d-ellipseList.txt" % i)
        print("reading FDDB-fold-%02d-ellipseList.txt ..." % i)
        with open(filename, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line[:-1] + ".jpg"
                img = cv2.imread(os.path.join(image_path, line))
                num_face = int(f.readline()[:-1])

                if img is None:
                    print("error, image: %s not found" % os.path.join(
                            image_path, line))
                    for _ in range(num_face):
                        f.readline()
                    continue

                for _ in range(num_face):
                    annot = f.readline().split(' ')
                    face_img_pos = crop(img, float(annot[3]), float(annot[4]),
                                        float(annot[0]), float(annot[1]),
                                        positive=True)
                    if i < 9:
                        train_pos.append(hog_skimage(face_img_pos))
                    else:
                        test_pos.append(hog_skimage(face_img_pos))

                    if i < 5 or i == 10:
                        face_img_neg = crop(img, float(annot[3]),
                                            float(annot[4]),
                                            float(annot[0]), float(annot[1]),
                                            positive=False)
                        if i < 5:
                            for j in range(9):
                                train_neg.append(hog_skimage(face_img_neg[j]))
                        else:
                            for j in range(9):
                                test_neg.append(hog_skimage(face_img_neg[j]))

    print("saving....")
    tpf = np.array(train_pos, dtype=np.float32)
    tpl = np.ones(tpf.shape[0], dtype=np.float32)
    np.savez(os.path.join(out_dir, "train_positive.npz"), feature=tpf,
             label=tpl)
    tnf = np.array(train_neg, dtype=np.float32)
    tnl = -1 * np.ones(tnf.shape[0], dtype=np.float32)
    np.savez(os.path.join(out_dir, "train_negative.npz"), feature=tnf,
             label=tnl)

    tp_n = np.concatenate((tpf, tpl.reshape(tpl.shape[0], 1)), axis=1)
    tn_n = np.concatenate((tnf, tnl.reshape(tnl.shape[0], 1)), axis=1)
    t_n = np.concatenate((tp_n, tn_n), axis=0)
    np.random.shuffle(t_n)
    np.savez(os.path.join(out_dir, "train.npz"), feature=t_n[:, :-1],
             label=t_n[:, -1])

    tpf = np.array(test_pos, dtype=np.float32)
    tpl = np.ones((tpf.shape[0], 1), dtype=np.float32)
    np.savez(os.path.join(out_dir, "test_positive.npz"), feature=tpf, label=tpl)

    tnf = np.array(test_neg, dtype=np.float32)
    tnl = -1 * np.ones(tnf.shape[0], dtype=np.float32)
    np.savez(os.path.join(out_dir, "test_negative.npz"), feature=tnf, label=tnl)

    tp_n = np.concatenate((tpf, tpl.reshape(tpl.shape[0], 1)), axis=1)
    tn_n = np.concatenate((tnf, tnl.reshape(tnl.shape[0], 1)), axis=1)
    t_n = np.concatenate((tp_n, tn_n), axis=0)
    np.random.shuffle(t_n)
    np.savez(os.path.join(out_dir, "test.npz"), feature=t_n[:, :-1],
             label=t_n[:, -1])


if __name__ == '__main__':
    get_hog_dataset()
#     fddb_path = os.path.join(DATA_PATH, 'FDDB-folds')
#     image_path = os.path.join(DATA_PATH, 'originalPics')
#     filename = os.path.join(fddb_path, "FDDB-fold-10-ellipseList.txt")
#     out_dir = os.path.join(DATA_PATH, "image_small")
#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
#
#     with open(filename, 'r') as f:
#         while True:
#             line = f.readline()
#             if not line:
#                 break
#             img_name = "_".join(line[:-1].split('/'))
#             line = line[:-1] + ".jpg"
#             img = cv2.imread(os.path.join(image_path, line))
#             num_face = int(f.readline()[:-1])
#             if img is None:
#                 print("error, image: %s not found" % os.path.join(
#                         image_path, line))
#                 for _ in range(num_face):
#                     f.readline()
#                 continue
#
#             # image exist
#             for _ in range(num_face):
#                 annot = f.readline().split(' ')
#                 face_img_pos = crop(img, float(annot[3]), float(annot[4]),
#                                 float(annot[0]), float(annot[1]),
#                                 positive=True)
#
#                 cv2.imwrite(
#                     os.path.join(out_dir, img_name + ".00.jpg"),
#                     face_img_pos)
#
#                 face_img_neg = crop(img, float(annot[3]),
#                                     float(annot[4]),
#                                     float(annot[0]),
#                                     float(annot[1]),
#                                     positive=False)
#
#                 for j in range(9):
#                     cv2.imwrite(os.path.join(out_dir,
#                                              img_name + ".%02d.jpg" % (
#                                                          j + 1)),
#                                 face_img_neg[j])
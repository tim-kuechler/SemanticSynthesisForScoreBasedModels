"""Script that was used to downscale the cityscapes dataset to 256p"""
import cv2
import os


def resize(root_dir, target_dir, height, width, label_or_inst):
    filenames = sorted(os.listdir(root_dir))

    for i, filename in enumerate(filenames):
        img = cv2.imread(os.path.join(root_dir, filename), cv2.IMREAD_UNCHANGED)
        if not label_or_inst:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        else:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(target_dir, filename), img)

        if i % 500 == 0:
            print('step: ', i)


if __name__ == '__main__':
    #train img
    resize('/export/data/tkuechle/datasets/cityscapes/train_img',
           '/export/data/tkuechle/datasets/cityscapes256/train_img',
           256, 512, False)
    #train labels
    resize('/export/data/tkuechle/datasets/cityscapes/train_label',
           '/export/data/tkuechle/datasets/cityscapes256/train_label',
           256, 512, True)
    #train inst
    resize('/export/data/tkuechle/datasets/cityscapes/train_inst',
           '/export/data/tkuechle/datasets/cityscapes256/train_inst',
           256, 512, True)
    #test label
    resize('/export/data/tkuechle/datasets/cityscapes/test_label',
           '/export/data/tkuechle/datasets/cityscapes256/test_label',
           256, 512, True)

    # test label colored
    resize('/export/data/tkuechle/datasets/cityscapes/test_label_color',
           '/export/data/tkuechle/datasets/cityscapes256/test_label_color',
           256, 512, True)

    #test inst
    resize('/export/data/tkuechle/datasets/cityscapes/test_inst',
           '/export/data/tkuechle/datasets/cityscapes256/test_inst',
           256, 512, True)
    # test img
    resize('/export/data/tkuechle/datasets/cityscapes/test_img',
           '/export/data/tkuechle/datasets/cityscapes256/test_img',
           256, 512, True)






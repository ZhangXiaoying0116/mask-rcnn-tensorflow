# note by xiaoying.zhang
every_k_epochs: step to save ckpt

# please change for bs32
_C.TRAIN.BATCH_SIZE_PER_GPU=32
'''
class ResNetFPNModel(DetectionModel):
    def __init__(self, fp16):
        super(ResNetFPNModel, self).__init__(fp16)

    def inputs(self):

        ret = [
            tf.placeholder(tf.string, (32,), 'filenames'), # N length vector of filenames
            tf.placeholder(tf.float32, (32, 512, 512, 3), 'images'),  # N x H x W x C
            tf.placeholder(tf.int32, (32, 3), 'orig_image_dims')  # N x 3(image dims - hwc)
        ]
'''

# please change for NMS_TOPK, which may inference fps
_C.RPN.TEST_PER_LEVEL_NMS_TOPK = 300 
_C.RPN.TEST_POST_LEVEL_NMS_TOPK = 64

# please change for 512*512 images input
--in data.py
im = cv2.resize(im, (512,512)) # in two place

# notice customer op : generate_bounding_box_proposals, roi_align
docker pull fewu/mask-rcnn-tensorflow:master-latest (tensorflow1.14 with 2 customer op above)

# if you want to create pb form ckpt, please run
python train.py --load ./train_log_bs32_300_64/ckpt/checkpoint --predict /home/alg/xiaoying.zhang/0910/tensorflow_models/models/research/object_detection/test_images/image1.jpg --output_pb ./train_log_bs32_300_64/ckpt/bs32_512_pre300_post64.pb
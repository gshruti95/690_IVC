subcommand = 'train' # change to 'resume' for ckpting or 'test' for eval
cuda = 1 # set it to 1 for running on GPU, 0 for CPU

content_image = './content-images/amber.jpg' # path to content img
style_image = './style-images/mosaic.jpg'# path to style-image
output_image = './amber-mosaic-2ep.jpg' # path to saved output img

dataset = './small_COCO/' # path to training dataset, the path should point to a folder containing another folder with all the training images
style_image_dir = './style-images/'
save_model_dir = './model2/' # path to folder where trained model will be saved
model = save_model_dir + '' # path to where model was saved
checkpoint_model_dir = './checkpoints2/' # path to folder where checkpoints of trained models will be saved
checkpoint_model = checkpoint_model_dir + 'ckpt_epoch_3.pth' # path to ckpt model to resume training

epochs = 7 # number of training epochs, default is 2
batch_size = 4 # batch size for training, default is 4
lr = 1e-3 # learning rate, default is 1e-3

num_styles = 7
content_scale = None # factor for scaling down the content img
content_weight = 1e5 # weight for content-loss, default is 1e5
style_weight = 1e10 # weight for style-loss, default is 1e10
style_size = None # size of style-image, default is the original size of style image
image_size = 256 # size of training images, default is 256 X 256

log_interval = 500 # number of images after which the training loss is logged, default is 500
checkpoint_interval = 1 # number of epochs after which a checkpoint of the trained model will be created, default 1

seed = 42 # random seed for training, default 42

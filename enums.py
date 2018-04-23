subcommand = 'train' # change to 'resume' for ckpting or anything else for eval
cuda = 1 # set it to 1 for running on GPU, 0 for CPU

content_image = './content-images/amber.jpg' # path to content img
#style_image = './style-images/candy.jpg'# path to style-image
output_image = './cond-1-model2' + '-2ep.jpg' # path to saved output img

s_idx = 1 # film - if None then eval mode, mix - needed, cond - needed
#s_list = [1,1,1,0,0,0,0,0,0] # film eval - if None then output is S images of diff styles
s_list = None
output_batch_dir = './col-film-'

spat = None # 1 for Ver , 0 for Hor
pre_color = 0 # if 1 preserve color of content image

dataset = './med_COCO/' # path to training dataset, the path should point to a folder containing another folder with all the training images
style_image_dir = './style-images/'
style_dataloader_dir = './style-loader/'
save_model_dir = './scomb-b3rep-ckpt-model/' # path to folder where trained model will be saved
checkpoint_model_dir = './scomb-b3rep-ckpt-checkpoints/' # path to folder where checkpoints of trained models will be saved
checkpoint_model = checkpoint_model_dir + 'ckpt_epoch_1.pth' # path to ckpt model to resume training
model = save_model_dir + 'epoch_2_Sun_Apr_22_17:38:06_2018_100000.0_10000000000.0.model' # path to where model was saved
#model = checkpoint_model

epochs = 2 # number of training epochs, default is 2
batch_size = 3 # batch size for training, default is 4
lr = 1e-3 # learning rate, default is 1e-3

num_styles = 9
content_scale = None # factor for scaling down the content img
content_weight = 1e5 # weight for content-loss, default is 1e5
style_weight = 1e10 # weight for style-loss, default is 1e10
style_size = None # size of style-image, default is the original size of style image
image_size = 256 # size of training images, default is 256 X 256

log_interval = 10 # number of batches after which the training loss is logged, default is 500
checkpoint_interval = 1 # number of epochs after which a checkpoint of the trained model will be created, default 1
batch_checkpoint_interval = 1500 # number of batches after which a checkpoint will be created in same dir as epoch checkpoints
seed = 42 # random seed for training, default 42

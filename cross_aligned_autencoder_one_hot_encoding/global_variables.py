# model parameters
N_ITER = 1000
BATCHLEN = 128
lambda_ = 0.1
y_dimension = 100
z_dimension = 100
discriminator_hidden_size = 100

# data path
path_to_data_folder = "../data"
train_positive_path = path_to_data_folder + "/sentiment.train.1.txt"
train_negative_path = path_to_data_folder + "/sentiment.train.0.txt"

w2v_path = path_to_data_folder + '/crawl-300d-200k.vec'

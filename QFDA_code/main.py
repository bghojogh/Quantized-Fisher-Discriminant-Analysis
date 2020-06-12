from my_QFDA import My_QFDA
from my_FDA import My_FDA
from my_kernel_FDA import My_kernel_FDA
import pickle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib import offsetbox
import pandas as pd
import scipy.io
import csv
import scipy.misc
import os
import math
from sklearn.model_selection import train_test_split   #--> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing   # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn.model_selection import KFold
from skimage.transform import rescale, resize


def main():
    # ---- settings:
    dataset = "Fashion_MNIST"  #--> ATT_glasses, MNIST, Fashion_MNIST
    kernel = "linear"  # --> ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’ --> if None, it is linear
    save_optimum_quantized_images = True
    do_validation = False
    load_previous_fit = True
    classify_KNN = False
    in_pixel_domain = False
    n_neighbors = 10
    if dataset == "MNIST" or dataset == "Fashion_MNIST":
        image_height, image_width = 28, 28
    manifold_learning_method = "QFDA" #--> QFDA, FDA, kernel_FDA
    search_method = "PSO"  #--> subgradient_method, exhaustive_search, PSO
    quantization_type = "uniform"  #--> uniform, non-uniform
    regularization_parameter1 = 1  #--> the
    regularization_parameter2 = 2
    n_bootstrap = 200
    split_to_train_and_test = True
    save_projection_directions_again = True
    reconstruct_using_howMany_projection_directions = None  # --> an integer >= 1, if None: using all "specified" directions when creating the python class
    process_out_of_sample_all_together = True
    project_out_of_sample = True
    n_projection_directions_to_save = 20 #--> an integer >= 1, if None: save all "specified" directions when creating the python class
    save_reconstructed_images_again = False
    save_reconstructed_outOfSample_images_again = False
    if dataset == "ATT_glasses":
        indices_reconstructed_images_to_save = None  #--> [100, 120]
        outOfSample_indices_reconstructed_images_to_save = None  #--> [100, 120]
    plot_projected_pointsAndImages_again = True
    which_dimensions_to_plot_inpointsAndImagesPlot = [0,1] #--> list of two indices (start and end), e.g. [1,3] or [0,1]
    subset_of_MNIST = True
    pick_subset_of_MNIST_again = False
    MNIST_subset_cardinality_training = 5000
    MNIST_subset_cardinality_testing = 1000

    if dataset == "ATT_glasses":
        path_dataset = "./Att_glasses/"
        n_samples = 400
        scale_factor = 2.0
        image_height = int(np.round(112 / scale_factor))
        image_width = int(np.round(92 / scale_factor))
        # print("image height: " + str(image_height) + ", image width: ", str(image_width))
        data = np.zeros((image_height * image_width, n_samples))
        labels = np.zeros((1, n_samples))
        image_index = -1
        for class_index in range(2):
            for filename in os.listdir(path_dataset + "class" + str(class_index+1) + "/"):
                image_index = image_index + 1
                if image_index >= n_samples:
                    break
                img = load_image(address_image=path_dataset + "class" + str(class_index+1) + "/" + filename)
                image_rescaled = rescale(img, 1.0 / scale_factor)  #--> https://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
                # image_rescaled = scipy.misc.imresize(arr=img, size=(1.0 / scale_factor))
                # print(img.shape)
                # print(image_rescaled.shape)
                # print(image_height)
                # print(image_width)
                # input("hi")
                # image_height = image_rescaled.shape[0]
                # image_width = image_rescaled.shape[1]
                # plt.imshow(image_rescaled, cmap='gray')
                # plt.colorbar()
                # plt.show()
                data[:, image_index] = image_rescaled.ravel()
                labels[:, image_index] = class_index
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- normalize (standardation):
        data_notNormalized = data
        # data = data / 255
        # scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
        # data = (scaler.transform(data.T)).T
        X_train = data
        Y_train = labels
        # ---- split into train, test, validation:
        # print(X_train.T.shape)
        # print(Y_train.ravel().shape)
        X_train, X_test, Y_train, Y_test = train_test_split(X_train.T, Y_train.ravel(), test_size=0.2, random_state=1, shuffle=True)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1, shuffle =True)
        X_train, X_test, X_val = X_train.T, X_test.T, X_val.T
        # Y_train, Y_test, Y_val = Y_train.reshape((1, -1)), Y_test.reshape((1, -1)), Y_val.reshape((1, -1))
        print("number of training images: " + str(X_train.shape[1]))
    elif dataset == 'MNIST':
        path_dataset_save = "./MNIST/"
        file = open(path_dataset_save+'X_train.pckl','rb')
        X_train = pickle.load(file); file.close()
        file = open(path_dataset_save+'y_train.pckl','rb')
        Y_train = pickle.load(file); file.close()
        Y_train = np.asarray(Y_train)
        file = open(path_dataset_save+'X_test.pckl','rb')
        X_test = pickle.load(file); file.close()
        file = open(path_dataset_save+'y_test.pckl','rb')
        Y_test = pickle.load(file); file.close()
        Y_test = np.asarray(Y_test)
        if subset_of_MNIST:
            if pick_subset_of_MNIST_again:
                X_train_picked = X_train[0:MNIST_subset_cardinality_training, :]
                X_test_picked = X_test[0:MNIST_subset_cardinality_testing, :]
                X_val_picked = X_train[MNIST_subset_cardinality_training:(MNIST_subset_cardinality_training+MNIST_subset_cardinality_testing), :]
                y_train_picked = Y_train[0:MNIST_subset_cardinality_training]
                y_test_picked = Y_test[0:MNIST_subset_cardinality_testing]
                y_val_picked = Y_train[MNIST_subset_cardinality_training:(MNIST_subset_cardinality_training+MNIST_subset_cardinality_testing)]
                save_variable(X_train_picked, 'X_train_picked', path_to_save=path_dataset_save)
                save_variable(X_test_picked, 'X_test_picked', path_to_save=path_dataset_save)
                save_variable(y_train_picked, 'y_train_picked', path_to_save=path_dataset_save)
                save_variable(y_test_picked, 'y_test_picked', path_to_save=path_dataset_save)
                save_variable(X_val_picked, 'X_val_picked', path_to_save=path_dataset_save)
                save_variable(y_val_picked, 'y_val_picked', path_to_save=path_dataset_save)
            else:
                file = open(path_dataset_save + 'X_train_picked.pckl', 'rb')
                X_train_picked = pickle.load(file)
                file.close()
                file = open(path_dataset_save + 'X_test_picked.pckl', 'rb')
                X_test_picked = pickle.load(file)
                file.close()
                file = open(path_dataset_save + 'y_train_picked.pckl', 'rb')
                y_train_picked = pickle.load(file)
                file.close()
                file = open(path_dataset_save + 'y_test_picked.pckl', 'rb')
                y_test_picked = pickle.load(file)
                file.close()
                file = open(path_dataset_save + 'X_val_picked.pckl', 'rb')
                X_val_picked = pickle.load(file)
                file.close()
                file = open(path_dataset_save + 'y_val_picked.pckl', 'rb')
                y_val_picked = pickle.load(file)
                file.close()
            X_train = X_train_picked
            X_test = X_test_picked
            X_val = X_val_picked
            Y_train = y_train_picked
            Y_test = y_test_picked
            Y_val = y_val_picked
        X_train = X_train.T
        X_test = X_test.T
        X_val = X_val.T
        # scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train.T)
        # X_train = (scaler.transform(X_train.T)).T
        # X_test = (scaler.transform(X_test.T)).T
        # X_val = (scaler.transform(X_val.T)).T
    elif dataset == "Fashion_MNIST":
        path_dataset_save = "./Fashion_MNIST/"
        if pick_subset_of_MNIST_again:
            with open(path_dataset_save+'fashion-mnist_train.csv', newline='') as csvfile:
                data = list(csv.reader(csvfile))
            data = np.asarray(data)
            data = data[1:, :]  #--> remove the first row which is names of columns
            data = data.astype(int)
            labels_train = data[:, 0].ravel()
            data = data[:, 1:]  # --> remove the first column which is the labels
            data_train = data.T  #--> data_train: rows are features and columns are samples
            with open(path_dataset_save+'fashion-mnist_test.csv', newline='') as csvfile:
                data2 = list(csv.reader(csvfile))
            data2 = np.asarray(data2)
            data2 = data2[1:, :]  #--> remove the first row which is names of columns
            data2 = data2.astype(int)
            labels_test = data2[:, 0].ravel()
            data2 = data2[:, 1:]  # --> remove the first column which is the labels
            data_test = data2.T  #--> data_train: rows are features and columns are samples
            # -- take subset of data:
            X_train = data_train[:, 0:MNIST_subset_cardinality_training]
            X_test = data_test[:, 0:MNIST_subset_cardinality_testing]
            X_val = data_train[:, MNIST_subset_cardinality_training:(MNIST_subset_cardinality_training+MNIST_subset_cardinality_testing)]
            Y_train = labels_train[0:MNIST_subset_cardinality_training]
            Y_test = labels_test[0:MNIST_subset_cardinality_testing]
            Y_val = labels_train[MNIST_subset_cardinality_training:(MNIST_subset_cardinality_training+MNIST_subset_cardinality_testing)]
            # -- save:
            save_variable(X_train, 'X_train', path_to_save=path_dataset_save)
            save_variable(X_test, 'X_test', path_to_save=path_dataset_save)
            save_variable(X_val, 'X_val', path_to_save=path_dataset_save)
            save_variable(Y_train, 'Y_train', path_to_save=path_dataset_save)
            save_variable(Y_test, 'Y_test', path_to_save=path_dataset_save)
            save_variable(Y_val, 'Y_val', path_to_save=path_dataset_save)
        else:
            file = open(path_dataset_save + 'X_train.pckl', 'rb')
            X_train = pickle.load(file)
            file.close()
            file = open(path_dataset_save + 'X_test.pckl', 'rb')
            X_test = pickle.load(file)
            file.close()
            file = open(path_dataset_save + 'X_val.pckl', 'rb')
            X_val = pickle.load(file)
            file.close()
            file = open(path_dataset_save + 'Y_train.pckl', 'rb')
            Y_train = pickle.load(file)
            file.close()
            file = open(path_dataset_save + 'Y_test.pckl', 'rb')
            Y_test = pickle.load(file)
            file.close()
            file = open(path_dataset_save + 'Y_val.pckl', 'rb')
            Y_val = pickle.load(file)
            file.close()
        # scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train.T)
        # X_train = (scaler.transform(X_train.T)).T
        # X_test = (scaler.transform(X_test.T)).T
        # X_val = (scaler.transform(X_val.T)).T

    if manifold_learning_method != "kernel_QFDA":
        path_to_save_base = "./output/" + dataset + "/" + manifold_learning_method
    else:
        path_to_save_base = "./output/" + dataset + "/" + manifold_learning_method + "/" + kernel

    # ---- save the optimum quantized images:
    if save_optimum_quantized_images:
        n_images_to_save_per_class = 20
        my_manifold_learning = My_QFDA(image_height=image_height, image_width=image_width, n_components=None, max_quantization_levels=255, quantization_type=quantization_type, search_method=search_method, regularization_parameter1=regularization_parameter1, regularization_parameter2=regularization_parameter2, n_bootstrap=n_bootstrap, in_pixel_domain=in_pixel_domain, kernel=kernel)
        path_to_save = path_to_save_base + "/reg_par1=" + str(regularization_parameter1) +", reg_par2=" + str(regularization_parameter2) + "/"
        X_quantized_inverse_dct_classes, X_padded_classes, X_padded_classes_meanAdded, padded_image_height, padded_image_width = my_manifold_learning.get_optimum_quantized_images(X=X_train, y=Y_train, path_to_save_base=path_to_save_base)
        labels_of_classes = list(set(Y_train))
        number_of_classes = len(labels_of_classes)
        for class_index in range(number_of_classes):
            X_quantized_inverse_dct_class = X_quantized_inverse_dct_classes[class_index]
            X_padded_class = X_padded_classes[class_index]
            X_padded_class_meanAdded = X_padded_classes_meanAdded[class_index]
            for sample_index in range(n_images_to_save_per_class):
                # ---- quantized:
                image = X_quantized_inverse_dct_class[:, sample_index]
                image = image.reshape(padded_image_height, padded_image_width)
                image = scipy.misc.imresize(arr=image, size=500)  # --> 5 times bigger
                save_image(image_array=image, path_without_file_name=path_to_save+"quantized/class"+str(class_index)+"/", file_name=str(sample_index)+".png")
                # ---- not quantized:
                image = X_padded_class[:, sample_index]
                image = image.reshape(padded_image_height, padded_image_width)
                # image = image.reshape(int(112 / 2), int(92 / 2))
                image = scipy.misc.imresize(arr=image, size=500)  # --> 5 times bigger
                save_image(image_array=image, path_without_file_name=path_to_save+"notQuantized/class"+str(class_index)+"/", file_name=str(sample_index)+".png")
                # ---- not quantized (mean added back):
                image = X_padded_class_meanAdded[:, sample_index]
                image = image.reshape(padded_image_height, padded_image_width)
                # image = image.reshape(int(112 / 2), int(92 / 2))
                image = scipy.misc.imresize(arr=image, size=500)  # --> 5 times bigger
                save_image(image_array=image, path_without_file_name=path_to_save+"notQuantized_mean/class"+str(class_index)+"/", file_name=str(sample_index)+".png")
        return

    # ---- validation:
    if do_validation:
        for regularization_parameter1 in [0.01, 0.1, 1, 10]:
            for regularization_parameter2 in [0.1, 0.5, 1, 1.5, 2]:
                print("****************** reg1 = " + str(regularization_parameter1) + ", reg2 = " + str(regularization_parameter2))
                my_manifold_learning = My_QFDA(image_height=image_height, image_width=image_width, n_components=None, max_quantization_levels=255, quantization_type=quantization_type, search_method=search_method, regularization_parameter1=regularization_parameter1, regularization_parameter2=regularization_parameter2, n_bootstrap=n_bootstrap, in_pixel_domain=in_pixel_domain, kernel=kernel)
                data_train_transformed, data_train_quantized_transformed = my_manifold_learning.fit_transform(X=X_train, y=Y_train, path_to_save_base=path_to_save_base, load_previous_fit=load_previous_fit)
                data_quantized_test_transformed = my_manifold_learning.transform(X=X_test, y=Y_test, transform_the_quantized=True)
                data_quantized_validation_transformed = my_manifold_learning.transform(X=X_val, y=Y_val, transform_the_quantized=True)
                path_to_save = path_to_save_base + "/reg_par1=" + str(regularization_parameter1) +", reg_par2=" + str(regularization_parameter2) + "/"
                data_test_transformed, data_validation_transformed = None, None
                KNN_classification(manifold_learning_method, data_train_quantized_transformed, data_quantized_test_transformed, data_quantized_validation_transformed,
                                   data_train_transformed, data_test_transformed, data_validation_transformed,
                                   Y_train, Y_test, Y_val, path_to_save, max_n_components=20, n_neighbors=n_neighbors)
        return

    # ---- fit + transform training data:
    if manifold_learning_method == "QFDA":
        my_manifold_learning = My_QFDA(image_height=image_height, image_width=image_width, n_components=None, max_quantization_levels=255, quantization_type=quantization_type, search_method=search_method, regularization_parameter1=regularization_parameter1, regularization_parameter2=regularization_parameter2, n_bootstrap=n_bootstrap, in_pixel_domain=in_pixel_domain, kernel=kernel)
        # data_train_transformed = my_manifold_learning.test_functions(X=X_train, y=Y_train, test_index=1)
        data_train_transformed, data_train_quantized_transformed = my_manifold_learning.fit_transform(X=X_train, y=Y_train, path_to_save_base=path_to_save_base, load_previous_fit=load_previous_fit)
    elif manifold_learning_method == "FDA":
        my_manifold_learning = My_QFDA(image_height=image_height, image_width=image_width, n_components=None, max_quantization_levels=255, quantization_type=quantization_type, search_method=search_method, regularization_parameter1=regularization_parameter1, regularization_parameter2=regularization_parameter2, n_bootstrap=n_bootstrap, in_pixel_domain=in_pixel_domain, kernel=kernel)
        data_train_transformed = my_manifold_learning.fit_transform_justFDA(X=X_train, y=Y_train, path_to_save_base=path_to_save_base, load_previous_fit=load_previous_fit, version=1)
        data_train_quantized_transformed = None
    elif manifold_learning_method == "kernel_FDA":
        my_manifold_learning = My_QFDA(image_height=image_height, image_width=image_width, n_components=None, max_quantization_levels=255, quantization_type=quantization_type, search_method=search_method, regularization_parameter1=regularization_parameter1, regularization_parameter2=regularization_parameter2, n_bootstrap=n_bootstrap, in_pixel_domain=in_pixel_domain, kernel=kernel)
        data_train_transformed = my_manifold_learning.fit_transform_justKernelFDA(X=X_train, y=Y_train, path_to_save_base=path_to_save_base, load_previous_fit=load_previous_fit)
        data_train_quantized_transformed = None

    # ---- transform out-of-sample data:
    if project_out_of_sample:
        if manifold_learning_method == "QFDA":
            data_quantized_test_transformed = my_manifold_learning.transform(X=X_test, y=Y_test, transform_the_quantized=True)
            data_quantized_validation_transformed = my_manifold_learning.transform(X=X_val, y=Y_val, transform_the_quantized=True)
            data_test_transformed, data_validation_transformed = None, None
        elif manifold_learning_method == "FDA":
            data_test_transformed = my_manifold_learning.transform_justFDA(X=X_test, y=Y_test)
            data_validation_transformed = my_manifold_learning.transform_justFDA(X=X_val, y=Y_val)
            data_quantized_test_transformed, data_quantized_validation_transformed = None, None
        elif manifold_learning_method == "kernel_FDA":
            data_test_transformed = my_manifold_learning.transform_justKernelFDA(X=X_test, y=Y_test)
            data_validation_transformed = my_manifold_learning.transform_justKernelFDA(X=X_val, y=Y_val)
            data_quantized_test_transformed, data_quantized_validation_transformed = None, None

    # ---- save projection directions:
    if save_projection_directions_again:
        path_to_save1 = path_to_save_base + "/reg_par1=" + str(regularization_parameter1) +", reg_par2=" + str(regularization_parameter2) + "/directions_dct/"
        path_to_save2 = path_to_save_base + "/reg_par1=" + str(regularization_parameter1) +", reg_par2=" + str(regularization_parameter2) + "/directions_idct/"
        print("Saving projection directions...")
        padded_image_height, padded_image_width = my_manifold_learning.get_padded_image_size()
        if manifold_learning_method == "QFDA":
            projection_directions, projection_directions_inverseDCT = my_manifold_learning.get_projection_directions(also_get_inverse_dct_of_U=True)
        elif manifold_learning_method == "FDA":
            projection_directions, projection_directions_inverseDCT = my_manifold_learning.get_projection_directions_justFDA(also_get_inverse_dct_of_U=True)
        if n_projection_directions_to_save == None:
            n_projection_directions_to_save = projection_directions.shape[1]
        for projection_direction_index in range(n_projection_directions_to_save):
            an_image = projection_directions[:, projection_direction_index].reshape((padded_image_height, padded_image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=500)  # --> 5 times bigger
            # save image:
            save_image(image_array=an_image, path_without_file_name=path_to_save1, file_name=str(projection_direction_index)+".png")
            an_image = projection_directions_inverseDCT[:, projection_direction_index].reshape((padded_image_height, padded_image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=500)  # --> 5 times bigger
            # save image:
            save_image(image_array=an_image, path_without_file_name=path_to_save2, file_name=str(projection_direction_index)+".png")

    # ---- save reconstructed images:
    if save_reconstructed_images_again:
        X_reconstructed = my_manifold_learning.reconstruct(X=X_train, scaler=None, using_howMany_projection_directions=reconstruct_using_howMany_projection_directions)
        if indices_reconstructed_images_to_save == None:
            indices_reconstructed_images_to_save = [0, X_reconstructed.shape[1]]
        for image_index in range(indices_reconstructed_images_to_save[0], indices_reconstructed_images_to_save[1]):
            an_image = X_reconstructed[:, image_index].reshape((image_height, image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=500)  # --> 5 times bigger
            # save image:
            if reconstruct_using_howMany_projection_directions is not None:
                tmp = "_using" + str(reconstruct_using_howMany_projection_directions) + "Directions"
            else:
                tmp = "_usingAllDirections"
            save_image(image_array=an_image, path_without_file_name="./output/"+dataset+"/"+manifold_learning_method+"/reconstructed_train"+tmp+"/", file_name=str(image_index)+".png")

    # Plotting the embedded data:
    if plot_projected_pointsAndImages_again:
        if dataset == "ATT_glasses":
            scale = 1
            dataset_notReshaped = np.zeros((n_samples, image_height*scale, image_width*scale))
            for image_index in range(n_samples):
                image = data_notNormalized[:, image_index]
                image_not_reshaped = image.reshape((image_height, image_width))
                image_not_reshaped_scaled = scipy.misc.imresize(arr=image_not_reshaped, size=scale*100)
                dataset_notReshaped[image_index, :, :] = image_not_reshaped_scaled
            if manifold_learning_method == "QFDA":
                fig, ax = plt.subplots(figsize=(10, 10))
                class_legends = ["no glasses, quantized", "glasses, quantized"]
                plot_components_by_colors(X_projected=data_train_quantized_transformed, y_projected=Y_train, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot, ax=None, markersize=8, class_legends=class_legends, colors=["b", "r"], markers=["o", "s"])
                # fig, ax = plt.subplots(figsize=(10, 10))
                # plot_componentsAndImages_2(X_projected=data_train_transformed, Y_projected=Y_train, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
                #                     images=255-dataset_notReshaped, ax=ax, image_scale=0.25, markersize=10, thumb_frac=0.01, cmap='gray_r')
            elif manifold_learning_method == "FDA" or manifold_learning_method == "kernel_FDA":
                fig, ax = plt.subplots(figsize=(10, 10))
                # ---- only take two dimensions to plot:
                # plot_components(X_projected=data_train_transformed, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
                #                     images=255-dataset_notReshaped, ax=ax, image_scale=0.25, markersize=10, thumb_frac=0.07, cmap='gray_r')
                class_legends = ["no glasses, original", "glasses, original"]
                # class_legends = None
                plot_components_by_colors(X_projected=data_train_transformed, y_projected=Y_train, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot, ax=None, markersize=8, class_legends=class_legends, colors=["b", "r"], markers=["o", "s"])
        elif dataset == "MNIST":
            class_legends = None
            if manifold_learning_method == "QFDA":
                plot_components_by_colors_withTest(X_train=data_train_quantized_transformed, X_test=data_quantized_test_transformed, y_train=Y_train, y_test=Y_test, which_dimensions_to_plot=[0,1], class_legends=class_legends, markersize=8, colors=None, markers=None)
            elif manifold_learning_method == "FDA" or manifold_learning_method == "kernel_FDA":
                plot_components_by_colors_withTest(X_train=data_train_transformed, X_test=data_test_transformed, y_train=Y_train, y_test=Y_test, which_dimensions_to_plot=[0,1], class_legends=class_legends, markersize=8, colors=None, markers=None)

    # classification (K nearest neighbor):
    if classify_KNN:
        path_to_save = path_to_save_base + "/reg_par1=" + str(regularization_parameter1) +", reg_par2=" + str(regularization_parameter2) + "/"
        KNN_classification(manifold_learning_method, data_train_quantized_transformed, data_quantized_test_transformed, data_quantized_validation_transformed,
                       data_train_transformed, data_test_transformed, data_validation_transformed,
                       Y_train, Y_test, Y_val, path_to_save, max_n_components=20, n_neighbors=n_neighbors)

def KNN_classification(manifold_learning_method, data_train_quantized_transformed, data_quantized_test_transformed, data_quantized_validation_transformed,
                       data_train_transformed, data_test_transformed, data_validation_transformed,
                       Y_train, Y_test, Y_val, path_to_save, max_n_components=20, n_neighbors=1):
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    error_test = np.zeros((max_n_components,))
    error_train = np.zeros((max_n_components,))
    error_validation = np.zeros((max_n_components,))
    for n_component in range(1,max_n_components+1):
        if manifold_learning_method == "QFDA":
            data_train_transformed_truncated = data_train_quantized_transformed[:n_component, :]
            # data_train_transformed_truncated = data_train_transformed[:n_component, :]
            data_test_transformed_truncated = data_quantized_test_transformed[:n_component, :]
            data_validation_transformed_truncated = data_quantized_validation_transformed[:n_component, :]
        elif manifold_learning_method == "FDA" or manifold_learning_method == "kernel_FDA":
            data_train_transformed_truncated = data_train_transformed[:n_component, :]
            data_test_transformed_truncated = data_test_transformed[:n_component, :]
            data_validation_transformed_truncated = data_validation_transformed[:n_component, :]
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(X=data_train_transformed_truncated.T, y=Y_train.ravel())
        predicted_Y_test = neigh.predict(X=data_test_transformed_truncated.T)
        error_test[n_component-1] = sum(predicted_Y_test != Y_test) / len(Y_test)
        predicted_Y_train = neigh.predict(X=data_train_transformed_truncated.T)
        error_train[n_component-1] = sum(predicted_Y_train != Y_train) / len(Y_train)
        predicted_Y_validation = neigh.predict(X=data_validation_transformed_truncated.T)
        error_validation[n_component - 1] = sum(predicted_Y_validation != Y_val) / len(Y_val)
        print("#components: "+str(n_component)+", error_validation: "+str(error_validation[n_component-1]))
    # validation error:
    error_validation_mean = error_validation.mean()
    error_validation_std = error_validation.std()
    error_validation_result = np.array([error_validation_mean, error_validation_std])
    print("average validation error: " + str(error_validation_mean) + " +- " + str(error_validation_std))
    # test error:
    error_test_mean = error_test.mean()
    error_test_std = error_test.std()
    error_test_result = np.array([error_test_mean, error_test_std])
    # train error:
    error_train_mean = error_train.mean()
    error_train_std = error_train.std()
    error_train_result = np.array([error_train_mean, error_train_std])
    # save:
    save_np_array_to_txt(variable=error_test, name_of_variable="error_test", path_to_save=path_to_save)
    save_np_array_to_txt(variable=error_train, name_of_variable="error_train", path_to_save=path_to_save)
    save_np_array_to_txt(variable=error_validation, name_of_variable="error_validation", path_to_save=path_to_save)
    save_np_array_to_txt(variable=error_validation_result, name_of_variable="error_validation_result", path_to_save=path_to_save)
    save_np_array_to_txt(variable=error_test_result, name_of_variable="error_test_result", path_to_save=path_to_save)
    save_np_array_to_txt(variable=error_train_result, name_of_variable="error_train_result", path_to_save=path_to_save)

def convert_mat_to_csv(path_mat, path_to_save):
    # https://gist.github.com/Nixonite/bc2f69b0c4430211bcad
    data = scipy.io.loadmat(path_mat)
    for i in data:
        if '__' not in i and 'readme' not in i:
            np.savetxt((path_to_save + i + ".csv"), data[i], delimiter=',')

def read_csv_file(path):
    # https://stackoverflow.com/questions/46614526/how-to-import-a-csv-file-into-a-data-array
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    # convert to numpy array:
    data = np.asarray(data)
    return data

def load_image(address_image):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    img_arr = np.array(img)
    return img_arr

def save_image(image_array, path_without_file_name, file_name):
    if not os.path.exists(path_without_file_name):
        os.makedirs(path_without_file_name)
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.fromarray(image_array)
    img = img.convert("L")
    img.save(path_without_file_name + file_name)

def show_image(img):
    plt.imshow(img)
    plt.gray()
    plt.show()

def plot_components_by_colors(X_projected, y_projected, which_dimensions_to_plot, class_legends=None, colors=None, markers=None, ax=None, markersize=10):
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    # plt.axis([0, 50, 60, 80])
    labels_of_classes = list(set(y_projected.ravel()))
    labels_of_classes.sort()  # --> sort ascending
    n_classes = len(labels_of_classes)
    if class_legends is None:
        class_legends_ = [""] * n_classes
    else:
        class_legends_ = class_legends
    if colors is None:
        colors_ = get_spaced_colors(n=n_classes)
    if markers is None:
        markers = ["o"] * n_classes
    for class_index in range(n_classes):
        class_label = labels_of_classes[class_index].astype(int)
        mask = (y_projected == class_label)
        mask = (mask == 1).ravel().tolist()
        X_projected_class = X_projected[mask, :]
        if colors is None:
            color = [colors_[class_index][0] / 255, colors_[class_index][1] / 255, colors_[class_index][2] / 255]
        else:
            color = colors[class_index]
        marker = markers[class_index]
        ax.plot(X_projected_class[:, 0], X_projected_class[:, 1], marker=marker, color=color, markersize=markersize, label=class_legends_[class_index], linestyle="None")
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    if class_legends is not None:
        ax.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_components_by_colors_withTest(X_train, X_test, y_train, y_test, which_dimensions_to_plot, class_legends=None, colors=None, markers=None, ax=None, markersize=10):
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_train = np.vstack((X_train[which_dimensions_to_plot[0], :], X_train[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_test = np.vstack((X_test[which_dimensions_to_plot[0], :], X_test[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_train = X_train.T
    X_test = X_test.T
    ax = ax or plt.gca()
    y_train = np.asarray(y_train)
    labels_of_classes = list(set(y_train.ravel()))
    labels_of_classes.sort()  # --> sort ascending
    n_classes = len(labels_of_classes)
    if class_legends is None:
        class_legends_ = [""] * n_classes
    else:
        class_legends_ = class_legends
    if colors is None:
        colors_ = get_spaced_colors(n=n_classes)
    if markers is None:
        markers = ["o"] * n_classes
    # plot training data:
    for class_index in range(n_classes):
        class_label = labels_of_classes[class_index].astype(int)
        mask = (y_train == class_label)
        mask = (mask == 1).ravel().tolist()
        X_train_class = X_train[mask, :]
        if colors is None:
            color = [colors_[class_index][0] / 255, colors_[class_index][1] / 255, colors_[class_index][2] / 255]
        else:
            color = colors[class_index]
        marker = markers[class_index]
        ax.plot(X_train_class[:, 0], X_train_class[:, 1], marker=marker, color=color, markersize=markersize, label="training, "+class_legends_[class_index], linestyle="None")
    # plot test data:
    for class_index in range(n_classes):
        class_label = labels_of_classes[class_index].astype(int)
        mask = (y_test == class_label)
        mask = (mask == 1).ravel().tolist()
        X_train_class = X_test[mask, :]
        if colors is None:
            color = [colors_[class_index][0] / 255, colors_[class_index][1] / 255, colors_[class_index][2] / 255]
        else:
            color = colors[class_index]
        marker = markers[class_index]
        ax.plot(X_train_class[:, 0], X_train_class[:, 1], marker=marker, color=color, markersize=markersize, label="test, "+class_legends_[class_index], linestyle="None", fillstyle='none')  #--> fillstyle = ('full', 'left', 'right', 'bottom', 'top', 'none')
    # plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    # plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    if class_legends is not None:
        ax.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()

def get_spaced_colors(n):
    # https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def plot_components(X_projected, which_dimensions_to_plot, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            ax.add_artist(imagebox)
        # # plot the first (original) image once more to be on top of other images:
        # # change color of frame (I googled: python OffsetImage highlight frame): https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

def plot_componentsAndImages_with_test(X_projected, X_test_projected, which_dimensions_to_plot, images=None, images_test=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    n_test_samples = X_test_projected.shape[1]
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_test_projected = np.vstack((X_test_projected[which_dimensions_to_plot[0], :], X_test_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    X_test_projected = X_test_projected.T
    ax = ax or plt.gca()
    ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
    ax.plot(X_test_projected[:, 0], X_test_projected[:, 1], '.r', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    images_test = resize(images_test, (images_test.shape[0], images_test.shape[1]*image_scale, images_test.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            # test image:
            if i < n_test_samples:
                dist = np.sum((X_test_projected[i] - shown_images) ** 2, 1)
                if np.min(dist) < min_dist_2:
                    # don't show points that are too close
                    continue
                shown_images = np.vstack([shown_images, X_test_projected[i]])
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images_test[i], cmap=cmap), X_test_projected[i], bboxprops =dict(edgecolor='red', lw=3))
                ax.add_artist(imagebox)
            # training image:
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            ax.add_artist(imagebox)
        # # plot the first (original) image once more to be on top of other images:
        # # change color of frame (I googled: python OffsetImage highlight frame):
        # https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # https://matplotlib.org/users/annotations_guide.html
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

def plot_componentsAndImages_2(X_projected, Y_projected, which_dimensions_to_plot, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    Y_projected = Y_projected.ravel()
    ax.plot(X_projected[Y_projected == 0, 0], X_projected[Y_projected == 0, 1], '.k', markersize=markersize)
    ax.plot(X_projected[Y_projected == 1, 0], X_projected[Y_projected == 1, 1], '.r', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            if Y_projected[i] == 0:
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            else:
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i], bboxprops =dict(edgecolor='red', lw=3))
            ax.add_artist(imagebox)
        # # plot the first (original) image once more to be on top of other images:
        # # change color of frame (I googled: python OffsetImage highlight frame): https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def read_facebook_dataset(path_dataset, sep=";", header='infer'):
    # return: X, Y --> rows: features, columns = samples
    data = pd.read_csv(path_dataset, sep=sep, header=header)
    # print(list(data.columns.values))
    names_of_input_features = ["Category", "Page total likes", "Type", "Post Month", "Post Hour", "Post Weekday", "Paid"]
    names_of_output_features = ["Lifetime Post Total Reach", "Lifetime Post Total Impressions", "Lifetime Engaged Users", "Lifetime Post Consumers", "Lifetime Post Consumptions", "Lifetime Post Impressions by people who have liked your Page", "Lifetime Post reach by people who like your Page", "Lifetime People who have liked your Page and engaged with your post", "comment", "like", "share", "Total Interactions"]
    X = np.zeros((data.shape[0], len(names_of_input_features)))
    for feature_index, feature_name in enumerate(names_of_input_features):
        try:
            X[:, feature_index] = data.loc[:, feature_name].values
        except:  # feature if categorical
            feature_vector = data.loc[:, feature_name]
            le = preprocessing.LabelEncoder()
            le.fit(feature_vector)
            X[:, feature_index] = le.transform(feature_vector)
    X = X.T
    Y = np.zeros((data.shape[0], len(names_of_output_features)))
    for feature_index, feature_name in enumerate(names_of_output_features):
        Y[:, feature_index] = data.loc[:, feature_name].values
    Y = Y.T
    # Five samples have some nan values, such as: the "Paid" feature of last sample (X[-1,-1]) is missing and thus nan. We remove it:
    indices_of_samples_not_having_missing_values = np.logical_and(~np.isnan(X).any(axis=0), ~np.isnan(Y).any(axis=0))  # https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-91.php
    X = X[:, indices_of_samples_not_having_missing_values]
    Y = Y[:, indices_of_samples_not_having_missing_values]
    return X, Y

def read_regression_dataset_withOneOutput(path_dataset, column_of_labels, sep=";", header='infer'):
    # return: X, Y --> rows: features, columns = samples
    data = pd.read_csv(path_dataset, sep=sep, header=header)
    X = np.zeros(data.shape)
    for feature_index in range(data.shape[1]):
        try:
            X[:, feature_index] = data.iloc[:, feature_index].values
        except:  # feature if categorical
            feature_vector = data.iloc[:, feature_index]
            le = preprocessing.LabelEncoder()
            le.fit(feature_vector)
            X[:, feature_index] = le.transform(feature_vector)
    Y = X[:, column_of_labels]
    X = np.delete(X, column_of_labels, 1)  # delete the column of labels from X
    X = X.T
    Y = Y.T
    return X, Y

if __name__ == '__main__':
    main()
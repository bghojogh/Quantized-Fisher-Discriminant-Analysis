import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_kernels
from my_generalized_eigen_problem import My_generalized_eigen_problem
from sklearn.preprocessing import MinMaxScaler   #--> https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
from my_generalized_eigen_problem import My_generalized_eigen_problem
import os
import pickle
from scipy import stats
import math
from scipy import fftpack
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  #--> https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
from scipy import optimize  #--> https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.anneal.html
import pso2


class My_QFDA:

    def __init__(self, image_height, image_width, n_components=None, max_quantization_levels=255, quantization_type="uniform", search_method="subgradient_method", regularization_parameter1=1, regularization_parameter2=1, n_bootstrap=100, in_pixel_domain=False, kernel="linear"):
        self.n_components = n_components
        self.U = None
        self.U_justFDA = None
        self.X_train = None
        self.y_train = None
        self.max_quantization_levels = max_quantization_levels
        self.quantization_type = quantization_type
        self.search_method = search_method
        self.S_T = None
        self.S_W = None
        self.S_T_justFDA = None
        self.S_W_justFDA = None
        self.regularization_parameter1 = regularization_parameter1
        self.regularization_parameter2 = regularization_parameter2
        self.image_height = image_height
        self.image_width = image_width
        self.block_size = 8
        self.padded_image_height = None
        self.padded_image_width = None
        self.max_of_range_array = None
        self.n_bootstrap = n_bootstrap
        self.bestCost = None
        self.best_number_of_levels = None
        self.kernel = kernel
        self.in_pixel_domain = in_pixel_domain
        self.X_padded_mean = None
        if search_method == "exhaustive_search":
            if quantization_type == "non-uniform":
                print("error: exhaustive_search is only for uniform quantization....")

    def divide_image_into_blocks(self, image):
        # image: with size (self.padded_image_height, self.padded_image_width)
        # blocks_reshaped_forImage: reshaped blocks stacked column-wise
        n_blocks_in_height = int(self.padded_image_height / self.block_size)
        n_blocks_in_width = int(self.padded_image_width / self.block_size)
        blocks_reshaped_forImage = np.zeros((self.block_size * self.block_size, n_blocks_in_height * n_blocks_in_width))
        counter = -1
        for block_index_in_height in range(n_blocks_in_height):
            for block_index_in_width in range(n_blocks_in_width):
                counter = counter + 1
                start_pixel_in_height = (block_index_in_height * self.block_size)
                end_pixel_in_height = (block_index_in_height * self.block_size) + self.block_size - 1
                start_pixel_in_width = (block_index_in_width * self.block_size)
                end_pixel_in_width = (block_index_in_width * self.block_size) + self.block_size - 1
                block = image[start_pixel_in_height:end_pixel_in_height + 1, start_pixel_in_width:end_pixel_in_width + 1]
                # plt.imshow(block, cmap='gray')
                # plt.colorbar()
                # plt.show()
                block_reshaped = block.reshape((-1, 1))
                blocks_reshaped_forImage[:, counter] = block_reshaped.ravel()
        return blocks_reshaped_forImage

    def concatenate_blocks_to_have_image(self, blocks):
        # blocks: stacked column-wise
        # image: with size (self.padded_image_height, self.padded_image_width)
        image = np.zeros((self.padded_image_height, self.padded_image_width))
        n_blocks_in_height = int(self.padded_image_height / self.block_size)
        n_blocks_in_width = int(self.padded_image_width / self.block_size)
        blocks_reshaped_forImage = np.zeros((self.block_size * self.block_size, n_blocks_in_height * n_blocks_in_width))
        counter = -1
        for block_index_in_height in range(n_blocks_in_height):
            for block_index_in_width in range(n_blocks_in_width):
                counter = counter + 1
                start_pixel_in_height = (block_index_in_height * self.block_size)
                end_pixel_in_height = (block_index_in_height * self.block_size) + self.block_size - 1
                start_pixel_in_width = (block_index_in_width * self.block_size)
                end_pixel_in_width = (block_index_in_width * self.block_size) + self.block_size - 1
                block_reshaped = blocks[:, counter]
                block = block_reshaped.reshape((self.block_size, self.block_size))
                # plt.imshow(block, cmap='gray')
                # plt.colorbar()
                # plt.show()
                image[start_pixel_in_height:end_pixel_in_height + 1, start_pixel_in_width:end_pixel_in_width + 1] = block
        return image

    def uniform_quantization(self, X_dct, n_quantization_levels_array, max_of_range_array):
        # X and return: columns are samples, rows are features
        # https://stackoverflow.com/questions/38152081/how-do-you-quantize-a-simple-input-using-python --> I searched: python quantization
        n_blocks_in_height = int(self.padded_image_height / self.block_size)
        n_blocks_in_width = int(self.padded_image_width / self.block_size)
        n_images = X_dct.shape[1]
        for image_index in range(n_images):
            image = X_dct[:, image_index].reshape((self.padded_image_height, self.padded_image_width))
            blocks_reshaped_forImage = self.divide_image_into_blocks(image=image)
            for frequency in range(self.block_size * self.block_size):
                n_quantization_levels = n_quantization_levels_array[frequency]
                max_of_range = max_of_range_array[frequency]
                DCT_values_for_that_frequency = blocks_reshaped_forImage[frequency, :]
                DCT_values_for_that_frequency[DCT_values_for_that_frequency >= max_of_range] = max_of_range
                DCT_values_for_that_frequency[DCT_values_for_that_frequency <= -1 * max_of_range] = -1 * max_of_range
                if n_quantization_levels % 2 == 0:  #--> if n_levels is even
                    temp = int((n_quantization_levels + 2) / 2)
                    temp2 = -1 * (temp - 2) * (max_of_range / temp)
                    DCT_values_for_that_frequency[DCT_values_for_that_frequency <= temp2] = temp2
                else: #--> if n_levels is odd
                    temp = int((n_quantization_levels + 1) / 2)
                DCT_values_for_that_frequency = np.sign(DCT_values_for_that_frequency) * (max_of_range / (temp - 1)) * np.floor(temp * np.abs(DCT_values_for_that_frequency) / max_of_range)
                blocks_reshaped_forImage[frequency, :] = DCT_values_for_that_frequency
            image = self.concatenate_blocks_to_have_image(blocks=blocks_reshaped_forImage)
            X_dct[:, image_index] = image.reshape((-1, 1)).ravel()
        X_dct_quantized = X_dct
        return X_dct_quantized

    def test_functions(self, X, y, test_index):
        self.X_train = X
        self.y_train = y
        np.random.seed(555)
        if test_index == 1:
            X_dct = self.DCT_blockwise(X=X)
            n_quantization_levels_array = 15.4 * np.ones((64,))
            # n_quantization_levels_array = 250 * np.ones((64,))
            n_quantization_levels_array = self.project_onto_constraint_set_integerValues(n_quantization_levels_array=n_quantization_levels_array)
            rate, max_of_range_array = self.calculate_rate(X_dct=X_dct, n_quantization_levels_array=n_quantization_levels_array)
            print(rate)
            X_dct_quantized = self.uniform_quantization(X_dct=X_dct, n_quantization_levels_array=n_quantization_levels_array, max_of_range_array=max_of_range_array)
            # print(X_dct_quantized.shape)
            # plt.imshow(X_dct_quantized[:, 0].reshape((self.padded_image_height, self.padded_image_width)), cmap='gray')
            # plt.colorbar()
            # plt.show()
            X_quantized_inverse_dct = self.inverse_DCT_blockwise(X_dct=X_dct, display_inverse_images=True)
            input("hi........")
        elif test_index == 2:
            n_quantization_levels_array = 2.3 * np.ones((64,))
            cost = self.cost_function(n_quantization_levels_array)
            print(cost)
            input("hi........")

    def get_optimum_quantized_images(self, X, y, path_to_save_base):
        X_ = X.copy()
        X_padded = self.zero_pad_if_necessary(X=X_)
        # ----------
        X_dct = self.DCT_blockwise(X=X)
        path_to_save = path_to_save_base + "/reg_par1=" + str(self.regularization_parameter1) +", reg_par2=" + str(self.regularization_parameter2) + "/"
        best_number_of_levels = self.load_variable(name_of_variable="best_number_of_levels", path=path_to_save)
        max_of_range_array = self.load_variable(name_of_variable="max_of_range_array", path=path_to_save)
        X_dct_quantized = self.uniform_quantization(X_dct=X_dct, n_quantization_levels_array=best_number_of_levels, max_of_range_array=max_of_range_array)
        X_quantized_inverse_dct = self.inverse_DCT_blockwise(X_dct=X_dct_quantized, display_inverse_images=False)
        # ----------
        X_quantized_inverse_dct = X_quantized_inverse_dct + self.X_padded_mean
        # ----------
        X_quantized_inverse_dct_classes = self._separate_samples_of_classes(X=X_quantized_inverse_dct, y=y)
        X_padded_meanAdded = X_padded + self.X_padded_mean
        X_padded_classes = self._separate_samples_of_classes(X=X_padded, y=y)
        X_padded_classes_meanAdded = self._separate_samples_of_classes(X=X_padded_meanAdded, y=y)
        return X_quantized_inverse_dct_classes, X_padded_classes, X_padded_classes_meanAdded, self.padded_image_height, self.padded_image_width

    def project_onto_constraint_set_integerValues(self, n_quantization_levels_array, consider_uppder_bound=True):
        # n_quantization_levels_array[n_quantization_levels_array > self.max_quantization_levels] = self.max_quantization_levels
        if consider_uppder_bound:
            mask = n_quantization_levels_array > self.max_of_range_array
            if np.any(mask == True):
                n_quantization_levels_array[mask == True] = self.max_of_range_array[mask == True]
        n_quantization_levels_array[n_quantization_levels_array < 2] = 2
        n_quantization_levels_array = np.ceil(n_quantization_levels_array - 0.5)
        return n_quantization_levels_array

    def cost_function(self, n_quantization_levels_array):
        # --- making n_levels integer:
        n_quantization_levels_array = self.project_onto_constraint_set_integerValues(n_quantization_levels_array=n_quantization_levels_array)
        # --- DCT of image blockwise:
        X_dct = self.DCT_blockwise(X=self.X_train)
        X_dct_ = X_dct.copy()
        # --- rate:
        rate, max_of_range_array = self.calculate_rate(X_dct=X_dct, n_quantization_levels_array=n_quantization_levels_array)
        rate_average = rate.mean()
        # --- quantized Fisher criterion:
        X_dct_quantized = self.uniform_quantization(X_dct=X_dct, n_quantization_levels_array=n_quantization_levels_array, max_of_range_array=max_of_range_array)
        if self.in_pixel_domain:
            X_idct = self.inverse_DCT_blockwise(X_dct=X_dct_, display_inverse_images=False)
            X_idct_quantized = self.inverse_DCT_blockwise(X_dct=X_dct_quantized, display_inverse_images=False)
            X_padded = self.zero_pad_if_necessary(X=self.X_train)
            self.FDA_with_quantization(X=X_padded, X_quantized=X_idct_quantized, y=self.y_train)
        else:
            self.FDA_with_quantization(X=X_dct_, X_quantized=X_dct_quantized, y=self.y_train)
        criterion = self.quantized_Fisher_criterion()
        # --- cost:
        cost = (-1 * criterion) + (self.regularization_parameter1 * rate_average)
        return cost

    def quantized_Fisher_criterion(self):
        numerator = np.trace((self.U.T).dot(self.S_T).dot(self.U))
        denominator = np.trace((self.U.T).dot(self.S_W).dot(self.U))
        criterion = numerator / denominator
        return criterion

    def FDA_with_quantization(self, X, X_quantized, y):
        # X, X_quantized --> columns are samples, rows are features
        # ------ Separate classes:
        X_separated_classes, X_quantized_separated_classes = self._separate_samples_of_classes_2(X=X, X_quantized=X_quantized, y=y)
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n_dimensions = X.shape[0]
        labels_of_classes = list(set(y.ravel()))
        n_classes = len(labels_of_classes)
        # ------ S_T:
        n = X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        # self.S_T = X.dot(H).dot(X.T) + X_quantized.dot(H).dot(X_quantized.T) + (2 * X.dot(H).dot(X_quantized.T))
        self.S_T = X_quantized.dot(H).dot(X_quantized.T) + (self.regularization_parameter2 * X.dot(H).dot(X_quantized.T))
        # self.S_T = X_quantized.dot(H).dot(X_quantized.T)
        # ------ S_W:
        self.S_W = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            # print("Calculating Sw: class " + str(class_index))
            X_class = X_separated_classes[class_index]
            X_quantized_class = X_quantized_separated_classes[class_index]
            n = X_class.shape[1]
            H = np.eye(n) - ((1 / n) * np.ones((n, n)))
            # self.S_W = self.S_W + ( X_class.dot(H).dot(X_class.T) + (self.regularization_parameter2 * X_quantized_class.dot(H).dot(X_quantized_class.T)) + (2 * X_class.dot(H).dot(X_quantized_class.T)) )
            self.S_W = self.S_W + ( (X_quantized_class.dot(H).dot(X_quantized_class.T)) + (self.regularization_parameter2 * X_class.dot(H).dot(X_quantized_class.T)) )
            # self.S_W = self.S_W + ( (X_quantized_class.dot(H).dot(X_quantized_class.T)) + (1 * X_quantized_class.dot(H).dot(X_class.T)) )
        # ------ Fisher directions:
        # print("Calculating eigenvectors...")
        # my_generalized_eigen_problem = My_generalized_eigen_problem(A=self.S_T, B=self.S_W)
        # eig_vec, eig_val = my_generalized_eigen_problem.solve()

        # eig_vec, eig_val = my_generalized_eigen_problem.solve_dirty()
        # print("Eigenvectors calculated.")
        # idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        # eig_val = eig_val[idx]
        # eig_vec = eig_vec[:, idx]

        # print("Calculating eigenvectors...")
        epsilon = 0.0000001  #--> to prevent singularity of matrix N
        eig_val, eig_vec = LA.eigh(inv(self.S_W + epsilon*np.eye(self.S_W.shape[0])).dot(self.S_T))
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]

        if self.n_components is not None:
            self.U = eig_vec[:, :self.n_components]
        else:
            self.U = eig_vec

    def kernel_FDA(self, X, y):
        # X: columns are sample, rows are features
        # ------ Separate classes:
        X_separated_classes = self._separate_samples_of_classes(X=X, y=y)
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n_samples = X.shape[1]
        labels_of_classes = list(set(y.ravel()))
        n_classes = len(labels_of_classes)

        # # ------ M_*:
        # Kernel_allSamples_allSamples = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        # M_star = Kernel_allSamples_allSamples.sum(axis=1)
        # M_star = M_star.reshape((-1, 1))
        # M_star = (1 / n_samples) * M_star
        # # ------ M_c and M:
        # M = np.zeros((n_samples, n_samples))
        # for class_index in range(n_classes):
        #     X_class = X_separated_classes[class_index]
        #     n_samples_of_class = X_class.shape[1]
        #     # ------ M_c:
        #     Kernel_allSamples_classSamples = pairwise_kernels(X=X.T, Y=X_class.T, metric=self.kernel)
        #     M_c = Kernel_allSamples_classSamples.sum(axis=1)
        #     M_c = M_c.reshape((-1, 1))
        #     M_c = (1 / n_samples_of_class) * M_c
        #     # ------ M:
        #     M = M + n_samples_of_class * (M_c - M_star).dot((M_c - M_star).T)
        # # ------ N:
        # N = np.zeros((n_samples, n_samples))
        # for class_index in range(n_classes):
        #     X_class = X_separated_classes[class_index]
        #     n_samples_of_class = X_class.shape[1]
        #     Kernel_allSamples_classSamples = pairwise_kernels(X=X.T, Y=X_class.T, metric=self.kernel)
        #     K_c = Kernel_allSamples_classSamples
        #     H_c = np.eye(n_samples_of_class) - (1 / n_samples_of_class) * np.ones(
        #         (n_samples_of_class, n_samples_of_class))
        #     N = N + K_c.dot(H_c).dot(K_c.T)
        # G_total = M
        # G_within = N


        # ------ G_total:
        print("hi1")
        Kernel_allSamples_allSamples = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        G_total = np.zeros((n_samples, n_samples))
        for sample_index in range(n_samples):
            g_sample = Kernel_allSamples_allSamples[:, sample_index]
            G_total = G_total + g_sample.dot(g_sample.T)
        print("hi2")
        # ------ G_within:
        G_within = np.zeros((n_samples, n_samples))
        for class_index in range(n_classes):
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            Kernel_allSamples_classSamples = pairwise_kernels(X=X.T, Y=X_class.T, metric=self.kernel)
            K_c = Kernel_allSamples_classSamples
            # H_c = np.eye(n_samples_of_class) - (1 / n_samples_of_class) * np.ones((n_samples_of_class, n_samples_of_class))
            G_within = G_within + K_c.dot(K_c.T)
        print("hi3")

        # ------ Fisher directions:
        # print("Calculating eigenvectors...")
        # my_generalized_eigen_problem = My_generalized_eigen_problem(A=G_total B=G_within)
        # eig_vec, eig_val = my_generalized_eigen_problem.solve()

        # eig_vec, eig_val = my_generalized_eigen_problem.solve_dirty()
        # print("Eigenvectors calculated.")
        # idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        # eig_val = eig_val[idx]
        # eig_vec = eig_vec[:, idx]

        my_generalized_eigen_problem = My_generalized_eigen_problem(A=G_total, B=G_within)
        eig_vec, eig_val = my_generalized_eigen_problem.solve()

        # print("Calculating eigenvectors...")
        # epsilon = 0.0000001  #--> to prevent singularity of matrix N
        # eig_val, eig_vec = LA.eigh(inv(G_within + epsilon*np.eye(G_within.shape[0])).dot(G_total))
        # idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        # eig_val = eig_val[idx]
        # eig_vec = eig_vec[:, idx]

        if self.n_components is not None:
            self.U_justFDA = eig_vec[:, :self.n_components]
        else:
            self.U_justFDA = eig_vec

    def FDA(self, X, y):
        # X --> columns are samples, rows are features
        # ------ Separate classes:
        X_separated_classes = self._separate_samples_of_classes(X=X, y=y)
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n_dimensions = X.shape[0]
        labels_of_classes = list(set(y.ravel()))
        n_classes = len(labels_of_classes)
        # ------ S_T:
        n = X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        self.S_T_justFDA = X.dot(H).dot(X.T)
        # ------ S_W:
        self.S_W_justFDA = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            # print("Calculating Sw: class " + str(class_index))
            X_class = X_separated_classes[class_index]
            n = X_class.shape[1]
            H = np.eye(n) - ((1 / n) * np.ones((n, n)))
            self.S_W_justFDA = self.S_W_justFDA + X_class.dot(H).dot(X_class.T)
        # ------ Fisher directions:
        # print("Calculating eigenvectors...")
        # my_generalized_eigen_problem = My_generalized_eigen_problem(A=self.S_T_justFDA, B=self.S_W_justFDA)
        # eig_vec, eig_val = my_generalized_eigen_problem.solve()

        # eig_vec, eig_val = my_generalized_eigen_problem.solve_dirty()
        # print("Eigenvectors calculated.")
        # idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        # eig_val = eig_val[idx]
        # eig_vec = eig_vec[:, idx]

        print("Calculating eigenvectors...")
        epsilon = 0.0000001  #--> to prevent singularity of matrix N
        eig_val, eig_vec = LA.eigh(inv(self.S_W_justFDA + epsilon*np.eye(self.S_W_justFDA.shape[0])).dot(self.S_T_justFDA))
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]

        if self.n_components is not None:
            self.U_justFDA = eig_vec[:, :self.n_components]
        else:
            self.U_justFDA = eig_vec

    def FDA_version2(self, X, y):
        # X: columns are sample, rows are features
        # self.X_train = X
        # ------ Separate classes:
        X_separated_classes = self._separate_samples_of_classes(X=X, y=y)
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n_samples = X.shape[1]
        n_dimensions = X.shape[0]
        labels_of_classes = list(set(y.ravel()))
        n_classes = len(labels_of_classes)
        # ------ S_W:
        self.S_W = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            # print("Calculating Sw: class " + str(class_index))
            X_class = X_separated_classes[class_index]
            n = X_class.shape[1]
            H = np.eye(n) - ((1 / n) * np.ones((n, n)))
            self.S_W = self.S_W + X_class.dot(H).dot(X_class.T)
        # ------ S_B:
        mean_of_total = X.mean(axis=1)
        mean_of_total = mean_of_total.reshape((-1, 1))
        S_B = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            mean_of_class = X_class.mean(axis=1)
            mean_of_class = mean_of_class.reshape((-1, 1))
            temp = mean_of_class - mean_of_total
            S_B = S_B + (n_samples_of_class * temp.dot(temp.T))
        # ------ M_c and M:
        S_W = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            print("Calculating Sw: class " + str(class_index))
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            mean_of_class = X_class.mean(axis=1)
            mean_of_class = mean_of_class.reshape((-1, 1))
            X_class_centered = X_class - mean_of_class
            for sample_index in range(n_samples_of_class):
                print("Calculating Sw: sample " + str(sample_index))
                temp = X_class_centered[:, sample_index]
                S_W = S_W + temp.dot(temp.T)
        # ------ Fisher directions:
        print("Calculating eigenvectors...")
        epsilon = 0.0000001  #--> to prevent singularity of matrix N
        eig_val, eig_vec = LA.eigh(inv(S_W + epsilon*np.eye(S_W.shape[0])).dot(S_B))
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        # my_generalized_eigen_problem = My_generalized_eigen_problem(A=S_B, B=S_W)
        # eig_vec, eig_val = my_generalized_eigen_problem.solve()
        # eig_vec, eig_val = my_generalized_eigen_problem.solve_dirty()
        if self.n_components is not None:
            U = eig_vec[:, :self.n_components]
        else:
            # U = eig_vec[:, :n_classes-1]
            U = eig_vec
        self.U_justFDA = U

    def calculate_max_of_ranges(self):
        n_quantization_levels_array = 15.4 * np.ones((64,))  #--> an example (not important what here)
        # --- making n_levels integer:
        n_quantization_levels_array = self.project_onto_constraint_set_integerValues(n_quantization_levels_array=n_quantization_levels_array, consider_uppder_bound=False)
        # --- DCT of image blockwise:
        X_dct = self.DCT_blockwise(X=self.X_train)
        # --- rate:
        rate, max_of_range_array = self.calculate_rate(X_dct=X_dct, n_quantization_levels_array=n_quantization_levels_array)
        return max_of_range_array

    def fit_transform(self, X, y, path_to_save_base, load_previous_fit=False):
        # X: columns are samples, rows are features
        self.fit(X=X, y=y, path_to_save_base=path_to_save_base, load_previous_fit=load_previous_fit)
        X_transformed = self.transform(X=X, y=y, transform_the_quantized=False)
        X_quantized_transformed = self.transform(X=X, y=y, transform_the_quantized=True)
        return X_transformed, X_quantized_transformed

    def fit_transform_justFDA(self, X, y, path_to_save_base, load_previous_fit=False, version=1):
        # X: columns are samples, rows are features
        self.fit_justFDA(X=X, y=y, path_to_save_base=path_to_save_base, load_previous_fit=load_previous_fit, version=version)
        X_transformed = self.transform_justFDA(X=X, y=y)
        return X_transformed

    def fit_transform_justKernelFDA(self, X, y, path_to_save_base, load_previous_fit=False):
        # X: columns are samples, rows are features
        self.fit_justKernelFDA(X=X, y=y, path_to_save_base=path_to_save_base, load_previous_fit=load_previous_fit)
        X_transformed = self.transform_justKernelFDA(X=X, y=y)
        return X_transformed

    def fit_justKernelFDA(self, X, y, path_to_save_base, load_previous_fit=False):
        self.X_train = X
        self.y_train = y
        path_to_save = path_to_save_base + "/reg_par1=" + str(self.regularization_parameter1) +", reg_par2=" + str(self.regularization_parameter2) + "/"
        if load_previous_fit:
            self.U_justFDA = self.load_variable(name_of_variable="U_justFDA", path=path_to_save)
            return
        X_dct = self.DCT_blockwise(X=self.X_train)
        self.kernel_FDA(X=X_dct, y=self.y_train)
        self.save_variable(self.U_justFDA, 'U_justFDA', path_to_save=path_to_save)
        # self.kernel_FDA(X=X, y=self.y_train)
        # self.save_variable(self.U_justFDA, 'U_justFDA', path_to_save=path_to_save)

    def fit_justFDA(self, X, y, path_to_save_base, load_previous_fit=False, version=1):
        self.X_train = X
        self.y_train = y
        path_to_save = path_to_save_base + "/reg_par1=" + str(self.regularization_parameter1) +", reg_par2=" + str(self.regularization_parameter2) + "/"
        if load_previous_fit:
            self.U_justFDA = self.load_variable(name_of_variable="U_justFDA", path=path_to_save)
            return
        X_dct = self.DCT_blockwise(X=self.X_train)
        if version == 1:
            self.FDA(X=X_dct, y=self.y_train)
        elif version == 2:
            self.FDA_version2(X=X_dct, y=self.y_train)
        self.save_variable(self.U_justFDA, 'U_justFDA', path_to_save=path_to_save)

    def fit(self, X, y, path_to_save_base, load_previous_fit=False):
        self.X_train = X
        self.y_train = y
        if load_previous_fit:
            path_to_save = path_to_save_base + "/reg_par1=" + str(self.regularization_parameter1) +", reg_par2=" + str(self.regularization_parameter2) + "/"
            self.U = self.load_variable(name_of_variable="U", path=path_to_save)
            self.best_number_of_levels = self.load_variable(name_of_variable="best_number_of_levels", path=path_to_save)
            self.max_of_range_array = self.load_variable(name_of_variable="max_of_range_array", path=path_to_save)
            return
        if self.search_method == "PSO":
            np.random.seed(555)  # Seeded to allow replication.
            # --- PSO 2 function:
            self.max_of_range_array = self.calculate_max_of_ranges()
            lower_bound = 2 * np.ones((self.block_size * self.block_size,))
            upper_bound = self.max_of_range_array
            a = pso2.PSO(objf=self.cost_function, lb=list(lower_bound), ub=list(upper_bound), dim=self.block_size*self.block_size, PopSize=5, iters=10)
            history_of_bestCosts = a.convergence
            self.bestCost = a.bestCost
            print("best cost: " + str(self.bestCost))
            self.best_number_of_levels = self.project_onto_constraint_set_integerValues(n_quantization_levels_array=a.bestParticle)
            print("best number of levels: " + str(self.best_number_of_levels))
        # --- train quantized Fisher subspace for the best quantization levels:
        X_dct = self.DCT_blockwise(X=self.X_train)
        X_dct_ = X_dct.copy()
        X_dct_quantized = self.uniform_quantization(X_dct=X_dct, n_quantization_levels_array=self.best_number_of_levels, max_of_range_array=self.max_of_range_array)
        if self.in_pixel_domain:
            X_idct = self.inverse_DCT_blockwise(X_dct=X_dct_, display_inverse_images=False)
            X_idct_quantized = self.inverse_DCT_blockwise(X_dct=X_dct_quantized, display_inverse_images=False)
            self.FDA_with_quantization(X=X_idct, X_quantized=X_idct_quantized, y=self.y_train)
            # xx = self.zero_pad_if_necessary(self.X_train)
            # # plt.imshow(xx[:, 0].reshape((self.padded_image_height, self.padded_image_width)), cmap='gray')
            # # plt.colorbar()
            # # plt.show()
            # self.FDA(X=xx, y=self.y_train)
        else:
            self.FDA_with_quantization(X=X_dct_, X_quantized=X_dct_quantized, y=self.y_train)
        # --- save variables:
        path_to_save = path_to_save_base + "/reg_par1=" + str(self.regularization_parameter1) +", reg_par2=" + str(self.regularization_parameter2) + "/"
        # path_to_save = "./output/" + str(self.search_method) +"/reg_par1=" + str(self.regularization_parameter1) +", reg_par2=" + str(self.regularization_parameter2) + "/"
        self.save_variable(self.U, 'U', path_to_save=path_to_save)
        self.save_variable(history_of_bestCosts, 'history_of_bestCosts', path_to_save=path_to_save)
        self.save_np_array_to_txt(history_of_bestCosts, 'history_of_bestCosts', path_to_save=path_to_save)
        self.save_variable(self.bestCost, 'bestCost', path_to_save=path_to_save)
        self.save_np_array_to_txt(self.bestCost, 'bestCost', path_to_save=path_to_save)
        self.save_variable(self.best_number_of_levels, 'best_number_of_levels', path_to_save=path_to_save)
        self.save_np_array_to_txt(self.best_number_of_levels, 'best_number_of_levels', path_to_save=path_to_save)
        self.save_variable(self.max_of_range_array, 'max_of_range_array', path_to_save=path_to_save)
        self.save_np_array_to_txt(self.max_of_range_array, 'max_of_range_array', path_to_save=path_to_save)

    def transform(self, X, y, transform_the_quantized=False):
        # X: columns are sample, rows are features
        # X_transformed: columns are sample, rows are features
        if transform_the_quantized:
            X_dct = self.DCT_blockwise(X=X)
            X_dct_quantized = self.uniform_quantization(X_dct=X_dct, n_quantization_levels_array=self.best_number_of_levels, max_of_range_array=self.max_of_range_array)
            if self.in_pixel_domain:
                X_idct_quantized = self.inverse_DCT_blockwise(X_dct=X_dct_quantized, display_inverse_images=False)
                X_transformed = (self.U.T).dot(X_idct_quantized)
            else:
                X_transformed = (self.U.T).dot(X_dct_quantized)
        else:
            if self.in_pixel_domain:
                X_padded = self.zero_pad_if_necessary(X=X)
                X_transformed = (self.U.T).dot(X_padded)
            else:
                X_dct = self.DCT_blockwise(X=X)
                X_transformed = (self.U.T).dot(X_dct)
        return X_transformed

    def transform_justFDA(self, X, y):
        # X: columns are sample, rows are features
        # X_transformed: columns are sample, rows are features
        # X_padded = self.zero_pad_if_necessary(X=X)
        X_dct = self.DCT_blockwise(X=X)
        X_transformed = (self.U_justFDA.T).dot(X_dct)
        return X_transformed

    def transform_justKernelFDA(self, X, y):
        # X: columns are sample, rows are features
        # X_transformed: columns are sample, rows are features
        # plt.imshow(self.X_train[:, 0].reshape((self.image_height, self.image_width)), cmap='gray')
        # plt.colorbar()
        # plt.show()
        X_train_dct = self.DCT_blockwise(X=self.X_train)
        X_dct = self.DCT_blockwise(X=X)
        Kernel_train_input = pairwise_kernels(X=X_train_dct.T, Y=X_dct.T, metric=self.kernel)
        X_transformed = (self.U_justFDA.T).dot(Kernel_train_input)
        # Kernel_train_input = pairwise_kernels(X=self.X_train.T, Y=X.T, metric=self.kernel)
        # X_transformed = (self.U_justFDA.T).dot(Kernel_train_input)
        return X_transformed

    def calculate_rate(self, X_dct, n_quantization_levels_array):
        # https://en.wikipedia.org/wiki/Quantization_(signal_processing)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.integrate_box_1d.html#scipy.stats.gaussian_kde.integrate_box_1d
        n_subset = self.n_bootstrap
        n_images = X_dct.shape[1]
        bootstrapped_images = np.random.choice(a=n_images, size=n_subset, replace=False)
        # X_dct_subset = X_dct[:, :n_subset]  #--> because of memory constraints (otherwise, gives memory error)
        X_dct_subset = X_dct[:, bootstrapped_images]  # --> because of memory constraints (otherwise, gives memory error)
        n_blocks_in_height = int(self.padded_image_height / self.block_size)
        n_blocks_in_width = int(self.padded_image_width / self.block_size)
        blocks_reshaped_forSubset = np.zeros((self.block_size*self.block_size, n_blocks_in_height*n_blocks_in_width*n_subset))
        for image_index in range(n_subset):
            image = X_dct_subset[:, image_index].reshape((self.padded_image_height, self.padded_image_width))
            blocks_reshaped_forImage = self.divide_image_into_blocks(image=image)
            blocks_reshaped_forSubset[:, (image_index*n_blocks_in_height*n_blocks_in_width) : ((image_index+1)*n_blocks_in_height*n_blocks_in_width)] = blocks_reshaped_forImage
        rate = np.zeros((self.block_size * self.block_size,))
        max_of_range_array = np.zeros((self.block_size * self.block_size,))
        for frequency in range(self.block_size * self.block_size):
            n_quantization_levels = n_quantization_levels_array[frequency]
            DCT_values_for_that_frequency = blocks_reshaped_forSubset[frequency, :].ravel()
            max_of_range = max(abs(DCT_values_for_that_frequency.min()), abs(DCT_values_for_that_frequency.max()))
            max_of_range_array[frequency] = max_of_range
            # print(DCT_values_for_that_frequency.mean())
            # print(DCT_values_for_that_frequency.std())
            # input("hiiiiiii")
            PDF = stats.gaussian_kde(dataset=blocks_reshaped_forSubset[frequency, :].ravel())
            entropy = 0
            if n_quantization_levels % 2 == 1:  # --> n_quantization_levels is odd
                temp = int((n_quantization_levels+1)/2)
            else:  #--> n_quantization_levels is even
                temp = int((n_quantization_levels+2)/2)
            for quantization_interval_index in range(temp):
                start1_of_interval = quantization_interval_index * (max_of_range / temp)
                end1_of_interval = (quantization_interval_index + 1) * (max_of_range / temp)
                start2_of_interval = -1 * quantization_interval_index * (max_of_range / temp)
                end2_of_interval = -1 * (quantization_interval_index + 1) * (max_of_range / temp)
                # if n_quantization_levels % 2 == 1:  #--> n_quantization_levels is odd
                #     start1_of_interval = quantization_interval_index * (max_of_range / temp)
                #     end1_of_interval = (quantization_interval_index+1) * (max_of_range / temp)
                #     start2_of_interval = -1 * quantization_interval_index * (max_of_range / temp)
                #     end2_of_interval = -1 * (quantization_interval_index+1) * (max_of_range / temp)
                # else:  #--> n_quantization_levels is even
                #     start1_of_interval = quantization_interval_index * (max_of_range / temp)
                #     end1_of_interval = (quantization_interval_index+1) * (max_of_range / temp)
                #     start2_of_interval = -1 * quantization_interval_index * (max_of_range / temp)
                #     end2_of_interval = -1 * (quantization_interval_index+1) * (max_of_range / temp)
                if quantization_interval_index == temp-1:  #--> last intervals
                    end1_of_interval = np.inf
                    end2_of_interval = -1 * np.inf
                if n_quantization_levels % 2 == 0:  #--> n_quantization_levels is even
                    if quantization_interval_index == temp-2:  #--> one to the last interval (for negative values)
                        end2_of_interval = -1 * np.inf
                # value_of_quantization = max_of_range * (quantization_interval_index / (temp - 1))
                p1 = PDF.integrate_box_1d(low=start1_of_interval, high=end1_of_interval)
                p2 = PDF.integrate_box_1d(low=end2_of_interval, high=start2_of_interval)
                p = p1 + p2
                if p != 0:
                    entropy = entropy + (-1 * p * math.log(p, 2.0))
            rate[frequency] = entropy
        return rate, max_of_range_array

    def DCT_blockwise(self, X):
        # X, X_dct: columns are samples, rows are features
        X_padded = self.zero_pad_if_necessary(X=X)
        n_blocks_in_height = int(self.padded_image_height / self.block_size)
        n_blocks_in_width = int(self.padded_image_width / self.block_size)
        X_dct = np.zeros(X_padded.shape)
        for image_index in range(X_padded.shape[1]):
            dct_of_image = np.zeros((self.padded_image_height, self.padded_image_width))
            for block_index_in_height in range(n_blocks_in_height):
                for block_index_in_width in range(n_blocks_in_width):
                    start_pixel_in_height = (block_index_in_height * self.block_size)
                    end_pixel_in_height = (block_index_in_height * self.block_size) + self.block_size - 1
                    start_pixel_in_width = (block_index_in_width * self.block_size)
                    end_pixel_in_width = (block_index_in_width * self.block_size) + self.block_size - 1
                    image = X_padded[:, image_index].reshape((self.padded_image_height, self.padded_image_width))
                    # plt.imshow(image, cmap='gray')
                    # plt.colorbar()
                    # plt.show()
                    block = image[start_pixel_in_height:end_pixel_in_height+1, start_pixel_in_width:end_pixel_in_width+1]
                    # plt.imshow(block, cmap='hot')
                    # plt.colorbar()
                    # plt.show()
                    block_dct = self.get_2D_dct(img=block)
                    # plt.imshow(block_dct, cmap='hot')
                    # plt.colorbar()
                    # plt.show()
                    dct_of_image[start_pixel_in_height:end_pixel_in_height+1, start_pixel_in_width:end_pixel_in_width+1] = block_dct
            X_dct[:, image_index] = dct_of_image.reshape((-1, 1)).ravel()
        return X_dct

    def inverse_DCT_blockwise(self, X_dct, display_inverse_images=False):
        # X_dct, X: columns are samples, rows are features
        n_blocks_in_height = int(self.padded_image_height / self.block_size)
        n_blocks_in_width = int(self.padded_image_width / self.block_size)
        X = np.zeros(X_dct.shape)
        for image_index in range(X_dct.shape[1]):
            image = np.zeros((self.padded_image_height, self.padded_image_width))
            for block_index_in_height in range(n_blocks_in_height):
                for block_index_in_width in range(n_blocks_in_width):
                    start_pixel_in_height = (block_index_in_height * self.block_size)
                    end_pixel_in_height = (block_index_in_height * self.block_size) + self.block_size - 1
                    start_pixel_in_width = (block_index_in_width * self.block_size)
                    end_pixel_in_width = (block_index_in_width * self.block_size) + self.block_size - 1
                    image_dct = X_dct[:, image_index].reshape((self.padded_image_height, self.padded_image_width))
                    # plt.imshow(image_dct, cmap='gray')
                    # plt.colorbar()
                    # plt.show()
                    block_dct = image_dct[start_pixel_in_height:end_pixel_in_height+1, start_pixel_in_width:end_pixel_in_width+1]
                    # plt.imshow(block_dct, cmap='hot')
                    # plt.colorbar()
                    # plt.show()
                    block = self.get_inverse_2D_dct(img_dct=block_dct)
                    # plt.imshow(block, cmap='gray')
                    # plt.colorbar()
                    # plt.show()
                    image[start_pixel_in_height:end_pixel_in_height+1, start_pixel_in_width:end_pixel_in_width+1] = block
            if display_inverse_images:
                plt.imshow(image, cmap='gray')
                plt.colorbar()
                plt.show()
            X[:, image_index] = image.reshape((-1, 1)).ravel()
        return X

    def get_2D_dct(self, img):
        # Get 2D Discrete Cosine Transform (DCT) of Image
        # http://bugra.github.io/work/notes/2014-07-12/discre-fourier-cosine-transform-dft-dct-image-compression/
        # return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')
        return fftpack.dct(fftpack.dct(img.T).T)

    def get_inverse_2D_dct(self, img_dct):
        # Get inverse 2D Discrete Cosine Transform (DCT) of Image
        # http://bugra.github.io/work/notes/2014-07-12/discre-fourier-cosine-transform-dft-dct-image-compression/
        # return fftpack.idct(fftpack.idct(img_dct.T, norm='ortho').T, norm='ortho')
        return fftpack.idct(fftpack.idct(img_dct.T, norm='ortho').T)

    def zero_pad_if_necessary(self, X):
        # X: columns are samples, rows are features
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.pad.html
        mod_for_height = self.image_height % self.block_size
        mod_for_height = self.block_size - mod_for_height
        if mod_for_height == 0:
            n_pad_before_for_height = 0
            n_pad_after_for_height = 0
        elif mod_for_height % 2 == 0:  # --> mod_for_height is even
            n_pad_before_for_height = int(mod_for_height / 2)
            n_pad_after_for_height = int(mod_for_height / 2)
        else:  # --> mod_for_height is odd
            n_pad_before_for_height = int(np.floor(mod_for_height / 2)) + 1
            n_pad_after_for_height = int(np.floor(mod_for_height / 2))
        mod_for_width = self.image_width % self.block_size
        mod_for_width = self.block_size - mod_for_width
        if mod_for_width == 0:
            n_pad_before_for_width = 0
            n_pad_after_for_width = 0
        elif mod_for_width % 2 == 0:  # --> mod_for_height is even
            n_pad_before_for_width = int(mod_for_width / 2)
            n_pad_after_for_width = int(mod_for_width / 2)
        else:  # --> mod_for_height is odd
            n_pad_before_for_width = int(np.floor(mod_for_width / 2)) + 1
            n_pad_after_for_width = int(np.floor(mod_for_width / 2))
        if n_pad_before_for_height == 0 and n_pad_after_for_height == 0 and n_pad_before_for_width == 0 and n_pad_after_for_width == 0:
            self.padded_image_height = self.image_height
            self.padded_image_width = self.image_width
            return X  #--> no pad
        X_padded = np.zeros(((self.image_height+mod_for_height) * (self.image_width+mod_for_width), X.shape[1]))
        for image_index in range(X.shape[1]):
            image = X[:, image_index].reshape((self.image_height, self.image_width))
            image_padded = np.lib.pad(image, ((n_pad_before_for_height, n_pad_after_for_height), (n_pad_before_for_width, n_pad_after_for_width)), 'constant', constant_values=0)
            # plt.imshow(image_padded, cmap='gray')
            # plt.colorbar()
            # plt.show()
            self.padded_image_height = image_padded.shape[0]
            self.padded_image_width = image_padded.shape[1]
            image_padded = image_padded.reshape((-1, 1))
            X_padded[:, image_index] = image_padded.ravel()
        #-- removing the mean:
        # X_padded = X_padded - 128
        self.X_padded_mean = X_padded.mean(axis=1).reshape((-1, 1))
        X_padded = X_padded - X_padded.mean(axis=1).reshape((-1,1))
        return X_padded

    def get_padded_image_size(self):
        return self.padded_image_height, self.padded_image_width

    def get_projection_directions(self, also_get_inverse_dct_of_U=False):
        if also_get_inverse_dct_of_U:
            U_inverse_dct = self.inverse_DCT_blockwise(X_dct=self.U, display_inverse_images=False)
        else:
            U_inverse_dct = None
        return self.U, U_inverse_dct

    def get_projection_directions_justFDA(self, also_get_inverse_dct_of_U=False):
        if also_get_inverse_dct_of_U:
            U_inverse_dct = self.inverse_DCT_blockwise(X_dct=self.U_justFDA, display_inverse_images=False)
        else:
            U_inverse_dct = None
        return self.U_justFDA, U_inverse_dct

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        return X_reconstructed

    def _separate_samples_of_classes(self, X, y):
        # X --> rows: features, columns: samples
        # X_separated_classes --> rows: features, columns: samples
        X = X.T
        y = np.asarray(y)
        y = y.reshape((-1, 1))
        yX = np.column_stack((y, X))
        yX = yX[yX[:, 0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
        y = yX[:, 0]
        X = yX[:, 1:]
        labels_of_classes = list(set(y))
        number_of_classes = len(labels_of_classes)
        dimension_of_data = X.shape[1]
        X_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
        class_index = 0
        index_start_new_class = 0
        n_samples = X.shape[0]
        for sample_index in range(1, n_samples):
            if y[sample_index] != y[sample_index - 1] or sample_index == n_samples-1:
                X_separated_classes[class_index] = np.vstack([X_separated_classes[class_index], X[index_start_new_class:sample_index, :]])
                index_start_new_class = sample_index
                class_index = class_index + 1
        for class_index in range(number_of_classes):
            X_class = X_separated_classes[class_index]
            X_separated_classes[class_index] = X_class.T
        return X_separated_classes

    def _separate_samples_of_classes_2(self, X, X_quantized, y):
        # X, X_quantized --> rows: features, columns: samples
        # X_separated_classes, X_quantized_separated_classes --> a list whose every element is --> rows: features, columns: samples
        X = X.T
        X_quantized = X_quantized.T
        y = np.asarray(y)
        y = y.reshape((-1, 1))
        yX = np.column_stack((y, X))
        yX = yX[yX[:, 0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
        y = yX[:, 0]
        X = yX[:, 1:]
        labels_of_classes = list(set(y))
        number_of_classes = len(labels_of_classes)
        dimension_of_data = X.shape[1]
        X_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
        X_quantized_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
        class_index = 0
        index_start_new_class = 0
        n_samples = X.shape[0]
        for sample_index in range(1, n_samples):
            if y[sample_index] != y[sample_index - 1] or sample_index == n_samples-1:
                X_separated_classes[class_index] = np.vstack([X_separated_classes[class_index], X[index_start_new_class:sample_index, :]])
                X_quantized_separated_classes[class_index] = np.vstack([X_quantized_separated_classes[class_index], X_quantized[index_start_new_class:sample_index, :]])
                index_start_new_class = sample_index
                class_index = class_index + 1
        for class_index in range(number_of_classes):
            X_class = X_separated_classes[class_index]
            X_separated_classes[class_index] = X_class.T
            X_class = X_quantized_separated_classes[class_index]
            X_quantized_separated_classes[class_index] = X_class.T
        return X_separated_classes, X_quantized_separated_classes

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable
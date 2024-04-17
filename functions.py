def estimate_pi(num_samples=30000000):
    import random
    points_inside_circle = 0

    for _ in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        distance_to_center = x**2 + y**2
        
        if distance_to_center <= 1:
            points_inside_circle += 1
    
    pi_estimate = 4 * points_inside_circle / num_samples
    return pi_estimate


def my_func(size=100):
    import numpy
    import pandas
    import sklearn.preprocessing
    import sklearn.decomposition
    import scipy.stats
    import scipy.special
    import matplotlib.pyplot
    import hashlib
    import os

    matrix_a = numpy.random.rand(int(size / 2), int(size / 2))
    matrix_b = numpy.random.rand(int(size / 2), int(size / 2))
    data = numpy.matmul(matrix_a, matrix_b)

    data = numpy.random.randn(size, 2) * 200 + numpy.random.rand(size, 2) * 100
    df = pandas.DataFrame(data, columns=['A', 'B'])
    df_normalized = df.apply(scipy.stats.zscore)
    scaler_standard = sklearn.preprocessing.StandardScaler().fit_transform(df_normalized)
    scaler_minmax = sklearn.preprocessing.MinMaxScaler().fit_transform(df_normalized)

    pca = sklearn.decomposition.PCA(n_components=2)
    pca_result = pca.fit_transform(scaler_standard)
    transformed_data = scipy.special.expit(pca_result)

    matplotlib.pyplot.figure(figsize=(12, 6))
    matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', label='PCA Result')
    matplotlib.pyplot.title("PCA Plot")

    unique_string = (str(transformed_data.tolist()) + str(pca_result.tolist()) + str(scaler_minmax.tolist()) + str(df_normalized.values.tolist()))
    hash_object = hashlib.sha256(unique_string.encode())
    hash_string = hash_object.hexdigest()

    # OMP_NUM_THREADS = os.environ.get('OMP_NUM_THREADS')

    return hash_string

def my_func_with_threadpoolctl(size=100):
    from threadpoolctl import threadpool_limits
    import numpy
    import pandas
    import sklearn.preprocessing
    import sklearn.decomposition
    import scipy.stats
    import scipy.special
    import matplotlib.pyplot
    import hashlib
    import os
    with threadpool_limits(limits=1):
        matrix_a = numpy.random.rand(int(size / 2), int(size / 2))
        matrix_b = numpy.random.rand(int(size / 2), int(size / 2))
        data = numpy.matmul(matrix_a, matrix_b)

        data = numpy.random.randn(size, 2) * 200 + numpy.random.rand(size, 2) * 100
        df = pandas.DataFrame(data, columns=['A', 'B'])
        df_normalized = df.apply(scipy.stats.zscore)
        scaler_standard = sklearn.preprocessing.StandardScaler().fit_transform(df_normalized)
        scaler_minmax = sklearn.preprocessing.MinMaxScaler().fit_transform(df_normalized)

        pca = sklearn.decomposition.PCA(n_components=2)
        pca_result = pca.fit_transform(scaler_standard)
        transformed_data = scipy.special.expit(pca_result)

        matplotlib.pyplot.figure(figsize=(12, 6))
        matplotlib.pyplot.subplot(1, 1, 1)
        matplotlib.pyplot.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', label='PCA Result')
        matplotlib.pyplot.title("PCA Plot")

        unique_string = (str(transformed_data.tolist()) + str(pca_result.tolist()) + str(scaler_minmax.tolist()) + str(df_normalized.values.tolist()))
        hash_object = hashlib.sha256(unique_string.encode())
        hash_string = hash_object.hexdigest()

        OMP_NUM_THREADS = os.environ.get('OMP_NUM_THREADS')

        if OMP_NUM_THREADS is not None:
            return OMP_NUM_THREADS
        else:
            return -1

def combine_csv(output_csv='resource_consumption_report.csv'):
    import pandas as pd
    import glob
    file_pattern = "task*.csv"
    task_files = glob.glob(file_pattern)

    combined_df = pd.DataFrame()
    for file in task_files:
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_csv, index=False)




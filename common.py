import os
import sys
import warnings
import numpy as np
from features import mfcc
import scipy.io.wavfile as wav
from sklearn import preprocessing
import six.moves.cPickle as pickle
from sklearn.cross_validation import train_test_split
from scipy.io.wavfile import write
import tensorflow as tf

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=wav.WavFileWarning)


def standardize_signal(signal):
    """
    Creates normalized amplitudes and Zero mean
    :param signal: a 1D list containing signal
    :return: a 1D list containing standardized signal
    """
    signal = (signal / float(max(abs(signal))))  # normalizo amplitude
    signal = signal - np.mean(signal)  # zero mean
    return signal


def read_audio(filepath):
    """
    Reads different types of audio files
    :param filepath: Full path to the audio file
    :return: 1D array with the raw signal data of the audio file
    """
    (rate, signal) = wav.read(filepath)
    return signal, rate


def standardize_features(feature_maps):
    """
    Center to the mean and component wise scale to unit variance
    :param feature_maps: Numpy array with dimensions (nr_excerpts, nr_frames, nr_features)
    :return: Standardized features with the same dimensions as input
    """
    new_feature_maps = np.array(feature_maps)
    samples = feature_maps.shape[0]
    length = feature_maps.shape[1]
    depth = feature_maps.shape[2]
    new_feature_maps = preprocessing.scale(new_feature_maps.reshape(samples, length * depth)).reshape(samples, length,
                                                                                                      depth)
    return new_feature_maps


def segment_signal(signal, rate, epsilon=0.101, alpha=0.5, output='mfcc', numcep=20, winlen=0.02, winstep=0.01):
    """
    Generates segments from audio signal by a algorithm by Juan Colonna.
    :param signal: 1D list of raw audio signal
    :param rate: Sample rate of the signal
    :param epsilon: The half of a width of the segment
    :param alpha: Amplitude threshold between 0~1
    :param output: 'mfcc' for MFCC features
    :param numcep: Number of coefficients
    :param winlen: Length of the frame in seconds
    :param winstep: Hop size of the frame window in seconds
    :return:
    """
    excerpts = []
    excerpts2 = []
    signal = standardize_signal(signal)

    # Zero padding
    n_points = int(round(epsilon * rate))
    signal = np.lib.pad(signal, (n_points, n_points), 'constant', constant_values=(0, 0))

    # Finds maximum and its index from signal list
    maximum, j = np.abs(signal).max(0), np.abs(signal).argmax(0)

    while maximum >= alpha:
        excerpt = np.array(signal[j - n_points:j + n_points])
        excerpt2 = np.empty((0,))
        if output == 'mfcc':
            excerpt2 = mfcc(excerpt, rate, numcep=numcep, winlen=(epsilon*2), winstep=(epsilon*2), nfilt=(2 * numcep))
            if excerpt2[0][1] == 0:
                print excerpt
            excerpt = mfcc(excerpt, rate, numcep=numcep, winlen=winlen, winstep=winstep, nfilt=(2 * numcep))

        excerpts.append(excerpt)
        excerpts2.append(excerpt2)

        #signal[j - n_points:j + n_points] = 0
        signal[j-n_points:j+n_points] = 0.01*np.random.randn(1, len(signal[j-n_points:j+n_points])); # fill the extracted values with random numbers (I prefer this option)
        maximum, j = np.abs(signal).max(0), np.abs(signal).argmax(0)  # Finds maximum and its index from signal list
    return np.array(excerpts), np.array(excerpts2)

def segment_files(dataset_dir, classes, epsilon=0.101, alpha=0.5, onehot=False, output='mfcc', numcep=20, winlen=0.02,
                  winstep=0.01, verbose=1, standardize=True):
    """
    Reads data and generates segments of it according to the algorithm created by .
    Each class should be in a separate directory.
    :param segment_by_samples: If true, the labels are output as [record_id, []]
    :param standardize: Scale to mean and unit variance
    :param dataset_dir: Path to a directory where class directories are located
    :param classes: Classes used in signal segmentation
    :param epsilon: The half of a width of the segment
    :param alpha: Amplitude threshold between 0~1
    :param onehot: if True returns class labels as one-hot encoded arrays, otherwise as INTs
    :param output: 'mfcc' for MFCC features
    :param numcep: Number of coefficients
    :param winlen: Length of the frame in seconds
    :param winstep: Hop size of the frame window in seconds
    :param verbose: 0 - doesn't show anything, 1 - shows the start and end of process, 2 - shows info for every class
                    processed
    :return: Segmented and standardized signal in 2D array of shape (nr_segments, 2xepsilon)
    """

    if verbose >= 1:
        print "\n\nStarting to preprocess data ..."
        sys.stdout.flush()
    excerpts = np.empty((0,))
    excerpts2 = np.empty((0,))
    labels = np.empty((0,))
    record_ids = np.empty((0,))
    class_index = 0
    record_id = 0

    for class_ in classes:
        class_excerpts = np.empty((0,))
        class_excerpts2 = np.empty((0,))
        class_labels = np.empty((0,))
        class_record_ids = np.empty((0,))
        if verbose >= 2:
            print '\nProcessing "' + class_ + '"'
            sys.stdout.flush()

        for filename in os.listdir(dataset_dir + class_ + '/'):

            sys.stdout.flush()
            filepath = dataset_dir + class_ + '/' + filename
            signal, rate = read_audio(filepath)
            segments, segments2 = segment_signal(signal, rate, epsilon, alpha, output, numcep, winlen, winstep)

            sample_labels = np.full((segments.shape[0],), class_index).astype(np.float32)
            sample_record_ids = np.full((segments.shape[0],), record_id).astype(np.float32)
            record_id += 1

            if class_excerpts.shape[0] == 0:
                class_excerpts = segments
                class_excerpts2 = segments2
                class_labels = sample_labels
                class_record_ids = sample_record_ids
            else:
                class_excerpts = np.concatenate((class_excerpts, segments))
                class_excerpts2 = np.concatenate((class_excerpts2, segments2))
                class_labels = np.concatenate((class_labels, sample_labels))
                class_record_ids = np.concatenate((class_record_ids, sample_record_ids))
            if verbose >= 3:
                print str(record_id) + '. ' + filename + ' processed. Dimensions (signal): ' + str(
                    segments.shape) + ' | ' + str(segments2.shape)

        class_index += 1
        if excerpts.shape[0] == 0:
            excerpts = class_excerpts
            excerpts2 = class_excerpts2
            labels = class_labels
            record_ids = class_record_ids
        else:
            excerpts = np.concatenate((excerpts, class_excerpts))
            excerpts2 = np.concatenate((excerpts2, class_excerpts2))
            labels = np.concatenate((labels, class_labels))
            record_ids = np.concatenate((record_ids, class_record_ids))

        if verbose >= 2:
            print 'Class "' + class_ + '" processed. Dimensions (excerpts | labels): ' + str(class_excerpts.shape) \
                  + ' | ' + str(class_labels.shape) + ' | ' + str(class_excerpts2.shape)
    if onehot:
        labels = (np.arange(len(classes)) == np.array(labels)[:, None]).astype(np.float32)

    if standardize and output != 'raw':
        excerpts = standardize_features(excerpts)

    if verbose >= 1:
        print '\nPreprocessing finished'
        print 'Excerpts shape: ' + str(excerpts.shape)
        print 'Excerpts2 shape: ' + str(excerpts2.shape)
        print 'Labels shape: ' + str(labels.shape)
        print 'Recording IDs shape: ' + str(record_ids.shape)
    return excerpts, excerpts2, labels, record_ids


def save_data(filename, excerpts, labels, record_ids=None, divide=True, testing_size=0.2, validation_size=0.2, verbose=1):
    """
    Saves data in pickle format
    :param record_ids:
    :param divide:
    :param filename:
    :param excerpts:
    :param labels:
    :param testing_size:
    :param validation_size:
    :param verbose:
    :return:
    """
    if divide and (record_ids is not None):
        training_data, testing_data, training_labels, testing_labels = train_test_split(excerpts, np.array(labels),
                                                                                        test_size=testing_size)

        training_data, validation_data, training_labels, validation_labels = train_test_split(
            training_data, training_labels, test_size=validation_size)
    else:
        training_data = excerpts
        training_labels = labels
        validation_data = np.empty((0,))
        validation_labels = np.empty((0,))
        testing_data = np.empty((0,))
        testing_labels = np.empty((0,))
    if verbose >= 2:
        print "Data before reshape"
        print "Training set: " + str(training_data.shape) + " | " + str(training_labels.shape)
        print "Validation set: " + str(validation_data.shape) + " | " + str(validation_labels.shape)
        print "Test set: " + str(testing_data.shape) + " | " + str(testing_labels.shape)

    training_data = reshape(training_data)
    validation_data = reshape(validation_data)
    testing_data = reshape(testing_data)

    if verbose >= 1:
        print "\nTraining set: " + str(training_data.shape) + " | " + str(training_labels.shape)
        print "Validation set: " + str(validation_data.shape) + " | " + str(validation_labels.shape)
        print "Test set: " + str(testing_data.shape) + " | " + str(testing_labels.shape)

    try:
        f = open(filename, 'wb')
        save = {
            'training_data': training_data,
            'training_labels': training_labels,
            'validation_data': validation_data,
            'validation_labels': validation_labels,
            'testing_data': testing_data,
            'testing_labels': testing_labels,
            'recording_ids' : record_ids
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        if verbose >= 1:
            print "Data saved!"
    except Exception as e:
        print('Unable to save data to', filename, ':', e)
        raise


def reshape(data, output_shape='tensorflow'):
    if data.shape[0] == 0:
        return data
    print data.shape
    segment_size = data.shape[1]
    mfcc_features = data.shape[2]
    num_channels = 1
    if output_shape == 'tensorflow':
        data = np.array(data).reshape(-1, segment_size, mfcc_features, num_channels).astype(np.float32)
    elif output_shape == 'theano':
        data = np.array(data).reshape(-1, num_channels, segment_size, mfcc_features).astype(np.float32)
    return data


def check_balance(classes, training_labels, validation_labels=None, testing_labels=None):
    # For checking the balance between the classes
    print 'Training set:'
    for i in range(len(classes)):
        print 'The class "' + classes[i] + '" has ' + str(len([x for x in training_labels if x == i])) + ' samples'
    if validation_labels is not None:
        print '\nValidation set:'
        for i in range(len(classes)):
            print 'The class "' + classes[i] + '" has ' + str(
                len([x for x in validation_labels if x == i])) + ' samples'

    if testing_labels is not None:
        print '\nTest set:'
        for i in range(len(classes)):
            print 'The class "' + classes[i] + '" has ' + str(len([x for x in testing_labels if x == i])) + ' samples'


def save_wav(filename, data):
    write(filename, 44100, np.int16(data * 32767))
    print filename + ' saved!'


def save_class_to_wav(class_name, dataset_dir, n_points=5000):
    excerpts, _ = segment_files(dataset_dir, [class_name], output='raw', verbose=1, standardize=True)
    signal = np.array([np.lib.pad(x, (n_points, n_points), 'constant', constant_values=(0, 0)) for x in excerpts])
    save_wav(class_name + '.wav', signal.flatten())
    print "File saved"


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def train_cnn_model(num_steps, batch_size, im_size, train_dataset, train_labels, test_dataset, valid_dataset=None, valid_labels=None,
                    num_channels=1, patch_size=3, depth=16, num_hidden=128, conv_stride=2, verbose=1,
                    seed=1337):
    num_labels = train_labels.shape[1]
    graph = tf.Graph()

    with graph.as_default():
        tf.set_random_seed(seed)
        # Input data.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, im_size, im_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        if valid_dataset is not None:
            tf_valid_dataset = tf.placeholder(tf.float32, shape=(valid_dataset.shape[0], im_size, im_size, num_channels))
        tf_test_dataset = tf.placeholder(tf.float32, shape=(test_dataset.shape[0], im_size, im_size, num_channels))

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        layer3_weights = tf.Variable(tf.truncated_normal(
            [im_size // 4 * im_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data):
            conv1 = tf.nn.conv2d(data, layer1_weights, [1, conv_stride, conv_stride, 1], padding='SAME')
            hidden1 = tf.nn.relu(conv1 + layer1_biases)
            conv2 = tf.nn.conv2d(hidden1, layer2_weights, [1, conv_stride, conv_stride, 1], padding='SAME')
            hidden2 = tf.nn.relu(conv2 + layer2_biases)
            shape = hidden2.get_shape().as_list()
            reshape = tf.reshape(hidden2, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

            if verbose >= 2:
                print 'Data shape: ' + str(data.get_shape())
                print 'Conv 1 shape: ' + str(conv1.get_shape())
                print 'Hidden 1 shape: ' + str(hidden1.get_shape())
                print 'Conv 2 shape: ' + str(conv2.get_shape())
                print 'Hidden 2 shape: ' + str(hidden2.get_shape())
                print 'Reshape: ' + str(reshape.get_shape())
                print 'Hidden 3 shape: ' + str(hidden3.get_shape())

            return tf.matmul(hidden3, layer4_weights) + layer4_biases

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        if valid_dataset is not None:
            valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

        if verbose >= 1:
            print 'Graph created. Model shape: ' + str(model(tf_train_dataset).get_shape())

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        if verbose >= 1:
            print '\nVariables initialized'
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, predictions = session.run(
                [optimizer, train_prediction], feed_dict=feed_dict)
            if step % 50 == 0:
                if verbose >= 2:
                    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                    if valid_dataset is not None:
                        print('Validation accuracy: %.1f%%' % accuracy(
                            valid_prediction.eval(feed_dict={tf_valid_dataset: valid_dataset}), valid_labels))

        test_predictions = test_prediction.eval(feed_dict={tf_test_dataset: test_dataset})
        if verbose >= 1:
            print '\n Training finished.'
    return test_predictions


def encode_onehot(nr_classes, labels):
    return (np.arange(nr_classes) == np.array(labels)[:, None]).astype(np.float32)

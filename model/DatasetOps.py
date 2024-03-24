import numpy as np
import tensorflow as tf
import scipy.io

def getDataset(
        base_path,
        buffer_size,
        batch_size):
    res101 = scipy.io.loadmat(base_path + 'res101.mat')
    all_features = np.transpose(res101['features'])
    all_labels = res101['labels'] - 1
    att_splits = scipy.io.loadmat(base_path + 'binaryAtt_splits.mat')
    classes_names = att_splits['allclasses_names']
    classes_names = [classes_names[i][0][0] for i in range(50)]
    class_attributes_matrix = att_splits['att']
    attributes_class_matrix = np.transpose(class_attributes_matrix)
    test_unseen = att_splits['test_unseen_loc'] - 1
    test_seen = att_splits['test_seen_loc'] - 1
    test = np.concatenate((test_unseen, test_seen))
    train = att_splits['trainval_loc'] - 1

    features_train = all_features[train].reshape((23527, 2048))
    labels_train = all_labels[train].reshape(23527)
    attributes_train = attributes_class_matrix[labels_train]
    features_test_unseen = all_features[test_unseen].reshape((7913, 2048))
    labels_test_unseen = all_labels[test_unseen].reshape(7913)
    attributes_test_unseen = attributes_class_matrix[labels_test_unseen]
    features_test_seen = all_features[test_seen].reshape((5882, 2048))
    labels_test_seen = all_labels[test_seen].reshape(5882)
    attributes_test_seen = attributes_class_matrix[labels_test_seen]
    features_test = all_features[test].reshape((13795, 2048))
    labels_test = all_labels[test].reshape(13795)
    attributes_test = attributes_class_matrix[labels_test]
    test_unseen_classes = np.unique(labels_test_unseen)
    train_classes = np.unique(labels_train)

    ds_train = tf.data.Dataset.from_tensor_slices((features_train, labels_train, attributes_train)).shuffle(
        buffer_size).batch(batch_size)
    ds_test_gzsl = tf.data.Dataset.from_tensor_slices((features_test, labels_test, attributes_test)).shuffle(
        buffer_size).batch(batch_size)
    ds_test_zsl = tf.data.Dataset.from_tensor_slices(
        (features_test_unseen, labels_test_unseen, attributes_test_unseen)).shuffle(buffer_size).batch(batch_size)

    all_data = {
        'attributes_class_matrix': attributes_class_matrix,
        'classes_names': classes_names,
        'features_train': features_train,
        'labels_train': labels_train,
        'attributes_train': attributes_train,
        'features_test': features_test,
        'labels_test': labels_test,
        'attributes_test': attributes_test,
        'features_test_seen': features_test_seen,
        'labels_test_seen': labels_test_seen,
        'attributes_test_seen': attributes_test_seen,
        'features_test_unseen': features_test_unseen,
        'labels_test_unseen': labels_test_unseen,
        'attributes_test_unseen': attributes_test_unseen,
        'test_unseen_classes': test_unseen_classes,
        'train_classes': train_classes
    }

    return ds_train, ds_test_zsl, ds_test_gzsl, all_data


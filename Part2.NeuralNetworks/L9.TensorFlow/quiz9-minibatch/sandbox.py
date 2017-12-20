# quiz
import math
import helper


def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    # pass
    size = int(len(features) / batch_size)

    batches = []
    for i in range(size):
        batches.append([features[i * batch_size:(i + 1) * batch_size],
                        labels[i * batch_size: (i + 1) * batch_size]])
    if len(features) % batch_size != 0:
        batches.append([features[size * batch_size:],
                        labels[size * batch_size:]])

    return batches


from pprint import pprint

# 4 Samples of features
example_features = [
    ['F11', 'F12', 'F13', 'F14'],
    ['F21', 'F22', 'F23', 'F24'],
    ['F31', 'F32', 'F33', 'F34'],
    ['F41', 'F42', 'F43', 'F44']]
# 4 Samples of labels
example_labels = [
    ['L11', 'L12'],
    ['L21', 'L22'],
    ['L31', 'L32'],
    ['L41', 'L42']]

# PPrint prints data structures like 2d arrays, so they are easier to read
pprint(batches(3, example_features, example_labels))

pprint(helper.batches(3, example_features, example_labels))

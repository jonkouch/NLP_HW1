# utils module
import numpy as np
import pickle
from preprocessing import represent_input_with_features, FeatureStatistics, Feature2id
from scipy.optimize import fmin_l_bfgs_b


def beam_search(sentence, pre_trained_weights, feature2id, beam_size=10):
    """
    Our implementation of beam search for MEMM
    """
    pi = {}
    bp = {}
    pi[0, '*', '*'] = 1
    bp[0, '*', '*'] = '*'
    found = False

    while not found:
        for k in range(1, len(sentence) + 1):
            for u in feature2id.feature_statistics.tags:
                for v in feature2id.feature_statistics.tags:
                    pi[k, u, v] = 0
                    bp[k, u, v] = None
                    for w in feature2id.feature_statistics.tags:
                        q = calc_q_value(feature2id, pre_trained_weights, sentence, k, u, v, w)
                        if q > pi[k, u, v]:
                            pi[k, u, v] = q
                            bp[k, u, v] = w
        if bp[len(sentence), '*', '*'] is not None:
            found = True
        else:
            beam_size -= 1
            if beam_size == 0:
                return None

        


def calc_q_value(feature2id, pre_trained_weights, sentence, k, u, v, w):
    """
    Calculates q value for beam search
    """
    curr_history = (sentence[k], u, sentence[k - 1], v, sentence[k - 2], w, sentence[k + 1])
    feature_dicts = feature2id.feature_to_idx
    feature_vector = represent_input_with_features(curr_history, feature_dicts)

    return np.dot(feature_vector, pre_trained_weights)
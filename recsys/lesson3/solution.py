from typing import List, Any

import numpy as np


def user_intersection(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> int:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: number of items in intersection of y_rel and y_rec (truncated to top-K)
    """
    return len(set(y_rec[:k]).intersection(set(y_rel)))


def user_hitrate(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> int:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: 1 if top-k recommendations contains at lease one relevant item
    """
    return int(user_intersection(y_rel, y_rec, k) > 0)


def user_precision(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of relevant items through recommendations
    """
    return user_intersection(y_rel, y_rec, k) / k


def user_recall(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of found relevant items through recommendations
    """
    return user_intersection(y_rel, y_rec, k) / len(set(y_rel))


def user_ap(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: average precision metric for user recommendations
    """
    return (
        np.sum(
            [
                user_precision(y_rel, y_rec, idx + 1)
                for idx, item in enumerate(y_rec[:k])
                if item in y_rel
            ]
        )
        / k
    )


def user_rr(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: reciprocal rank for user recommendations
    """
    for idx, item in enumerate(y_rec[:k]):
        if item in y_rel:
            return 1 / (idx + 1)
    return 0


def user_ndcg(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: ndcg metric for user recommendations
    """
    dcg = sum(
        [1.0 / np.log2(idx + 2) for idx, item in enumerate(y_rec[:k]) if item in y_rel]
    )
    idcg = sum(
        [1.0 / np.log2(idx + 2) for idx, _ in enumerate(zip(y_rel, np.arange(k)))]
    )
    return dcg / idcg

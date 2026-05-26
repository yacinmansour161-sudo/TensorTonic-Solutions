def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    a=0
    for i in range (k):
        if recommended[i] in relevant:
            a+=1
    return [a/k , a/len(relevant)]
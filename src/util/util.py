def X_y_split(df, target):
    """
    """
    X = df.drop(target, axis=1)
    # y = df[target] # output y as a Series
    y = df[[target]] # output y as a DataFrame
    return X, y


def split_range_into_k_segments(start, end, k):
    """
    Splits a range [a, b] into k equal segments with integer boundaries.
    
    Parameters:
        a (int): Start of the range.
        b (int): End of the range.
        k (int): Number of segments.
    
    Returns:
        list: A list of tuples, where each tuple represents an integer segment (start, end).
    """
    if k <= 0:
        raise ValueError("Number of segments (k) must be greater than zero.")
    if start > end:
        raise ValueError("Start of the range (a) must be less than or equal to the end (b).")
    
    # Compute approximate segment size
    segment_size = (end - start) // k
    remainder = (end - start) % k
    
    # Create segments
    segments = []
    current_start = start
    for i in range(k):
        current_end = current_start + segment_size
        if remainder > 0:
            current_end += 1
            remainder -= 1
        # Adjust the last segment to end exactly at b
        if i == k - 1:
            current_end = end
        segments.append((current_start, current_end))
        current_start = current_end
    
    return segments
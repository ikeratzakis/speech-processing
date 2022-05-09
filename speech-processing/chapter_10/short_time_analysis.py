# Function to calculate the shortTimeEnergy for given windowLength and step.
# This is a direct python translation from the MATLAB file found in the e-class.
# Code from sliding is from https://gist.github.com/fasiha/957035272009eb1c9eb370936a6af2eb
def short_time_energy(signal, window_length, step):
    cur_pos = 0
    length = len(signal)
    energy = []
    # frameCounter = 1
    while cur_pos + window_length - 1 <= length:
        window = signal[cur_pos:cur_pos + window_length - 1]
        window_energy = (1 / window_length) * sum([x ** 2 for x in window])
        energy.append(window_energy)
        cur_pos += step
        # frameCounter +=1

    return energy


def sliding(x, Nwin, Noverlap=0, f=lambda x: x):
    """Apply a function over overlapping sliding windows with overlap.

  Given an iterator with N elements (a list, a Numpy vector, a range object,
  etc.), subdivide it into Nwin-length chunks, potentially with Noverlap samples
  overlapping between chunks. Optionally apply a function to each such chunk.

  Any chunks at the end of x whose length would be < Nwin are silently ignored.

  Parameters
  ----------
  x : array_like
      Iterator (list, vector, range object, etc.) to operate on.
  Nwin : int
      Length of each chunk (sliding window).
  Noverlap : int, optional
      Amount of overlap between chunks. Noverlap must be < Nwin. 0 means no
      overlap between chunks. Positive Noverlap < Nwin means the last Noverlap
      samples of a chunk will be the first Noverlap samples of the next chunk.
      Negative Noverlap means |Noverlap| samples of the input will be skipped
      between each successive chunk.
  f : function, optional
      A function to apply on each chunk. The default is the identity function
      and will just return the chunks.

  Returns
  -------
  l : list
      A list of chunks, with the function f applied.
  """
    hop = Nwin - Noverlap
    return [f(x[i: i + Nwin])
            for i in range(0, len(x), hop)
            if i + Nwin <= len(x)]


# Function to calculate the shortTimeZeroCrossingRate for given windowLength and step.
# This is a direct python translation from the MATLAB file found in the e-class.

def zero_crossing_rate(signal, window_length, step):
    cur_pos = 0
    length = len(signal)
    zcr = []
    # frameCounter = 1
    while cur_pos + window_length - 1 <= length:
        window = signal[cur_pos:cur_pos + window_length - 1]

        temp = 0
        for i in range(1, window_length - 1):
            if window[i] >= 0 > window[i - 1]:
                temp += 2
            if window[i] < 0 <= window[i - 1]:
                temp += 2
        window_zcr = (1 / (2 * window_length)) * temp
        zcr.append(window_zcr)
        cur_pos += step
        # frameCounter +=1
    return zcr

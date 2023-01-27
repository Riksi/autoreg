import numpy as np


def get_mask(kernel_size, features_in, features_out, mode='B'):
    assert (features_in % 3) == 0, features_in
    assert (features_out % 3) == 0, features_out

    # Assume centre of kernel corresponds to present pixel
    # which will be the case if kernel dims are odd
    # and "same" padding is used
    h, w = kernel_size
    i_cent = h // 2
    j_cent = w // 2
    mask = np.ones((h, w, features_in, features_out))

    # all zeros to the left in the centre row
    mask[i_cent, (j_cent + 1):] = 0.
    # all zeros below
    mask[(i_cent + 1):] = 0.

    # clr2clr[i, j] indicates whether colour_i
    # in the previous layer is connected to colour_j in present
    # where colours are R,G,B.
    # Entries above the diagonal are always 1 and below always 0
    # since there is no backward flow.
    # For mask B a colour feeds into itself in the next layer
    # so the diagonal entries are 1 but for mask A they are 0
    # meaning that the colour can't feed into itself if mask A is used
    clr2clr = np.triu(np.ones((3, 3)), k=1 if mode == 'A' else 0)

    rgb_mask = np.repeat(np.repeat(clr2clr, features_in // 3, axis=0),
                         features_out // 3, axis=1)

    mask[i_cent, j_cent] = rgb_mask

    return mask.astype('float32')
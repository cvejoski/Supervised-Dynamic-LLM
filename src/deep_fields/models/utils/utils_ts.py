import torch


def unfold_steps_ahead(series,steps_ahead):
    """
    :param series: torch tensor [btahc_size,sequence_lenght,dimension]
    :param steps_ahead: how much to unfold
    :return: [batch_size * sequence_lenght_,steps_ahead]
    """
    batch_size = series.shape[0]
    sequence_lenght = series.shape[1]
    dimension = series.shape[2]

    sequence_lenght_ = sequence_lenght - steps_ahead + 1
    unfolded_series = series.unfold(dimension=1, size=steps_ahead, step=1).contiguous()
    unfolded_series = unfolded_series.reshape(batch_size * sequence_lenght_,
                                              dimension,
                                              steps_ahead)
    unfolded_series = unfolded_series.permute(0, 2, 1)
    # past = unfolded_series[:, :-1, :]
    # future = unfolded_series[:, 1:, :]
    return unfolded_series
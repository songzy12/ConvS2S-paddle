def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception(
                "invalid number of parameters in convolution spec "
                + str(spec)
                + ". expected 2 or 3"
            )
    return tuple(extended)

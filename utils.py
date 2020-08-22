def get_fc_input_size(max_seq_len):
        """
        Returns the value of input_features parameter for the first Linear layer
        @params max_seq_len (int): Maximum number of characters considered for
        input
        @returns x (int): The input_features parameter for first Linear layer
        """
        x = conv_output(max_seq_len,7,0,1)
        x = conv_output(x,3,0,3)
        x = conv_output(x,7,0,1)
        x = conv_output(x,3,0,3)
        x = conv_output(x,3,0,1)
        x = conv_output(x,3,0,1)
        x = conv_output(x,3,0,1)
        x = conv_output(x,3,0,1)
        x = conv_output(x,3,0,3)
        return x

def conv_output(input_size,kernel_size,padding_size,stride):
    """
    Returns the output sequence length after a  1d convolution/max pooling
    operation according to the formula
    output_size = floor(input_size - 2*padding_size - kernel_size)/stride+1
    @params input_size (int): Length of input sequence
    @params kernel_size (int): number of time steps that the kernel
    convolves over
    @params padding_size (int): number of pixels used to pad the input on
    one size of the input matrix
    @params stride (int): Kernel stride
    @returns Output sequence length
    """
    return (input_size-2*padding_size-kernel_size)//stride + 1

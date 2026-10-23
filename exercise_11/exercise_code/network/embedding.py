from torch import nn
import torch

def positional_encoding(d_model: int,
                        max_length: int) -> torch.Tensor:
    """
    Computes the positional encoding matrix
    Args:
        d_model: Dimension of Embedding
        max_length: Maximums sequence length

    Shape:
        - output: (max_length, d_model)
    """

    output = None

    ########################################################################
    # TODO:                                                                #
    #   Task 4: Initialize the positional encoding layer.                  #
    #                                                                      #
    # Hints 4:                                                             #
    #       - You can copy the implementation from the notebook, just      #
    #         make sure to use torch instead of numpy!                     #
    #       - Use torch.log(torch.Tensor([10000])), to make use of the     #
    #         torch implementation of the natural logarithm.               #
    #       - Implement the alternating sin and cos functions the way we   #
    #         did in the notebook.                                         #
    ########################################################################


    if d_model <= 0 or max_length < 0:
        raise ValueError("d_model must be positive and max_length non-negative")

    positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / d_model)
    )
    angles = positions * frequencies
    output = torch.zeros(max_length, d_model, dtype=torch.float32)
    output[:, 0::2] = torch.sin(angles)
    # An odd embedding width has one more sine channel than cosine channel.
    output[:, 1::2] = torch.cos(angles[:, :d_model // 2])

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return output

class Embedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_length: int):
        """

        Args:
            vocab_size: Number of elements in the vocabulary
            d_model: Dimension of Embedding
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embedding = None
        self.pos_encoding = None

        ########################################################################
        # TODO:                                                                #
        #   Task 1: Initialize the embedding layer (torch.nn implementation)   #
        #   Task 4: Initialize the positional encoding layer.                  #
        #                                                                      #
        # Hints 1:                                                             #
        #       - Have a look at pytorch embedding module                      #
        # Hints 4:                                                             #
        #       - Initialize it using d_model and max_length                   #
        ########################################################################


        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(d_model, max_length)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        # We will convert it into a torch parameter module for you! You can treat it like a normal tensor though!
        if self.pos_encoding is not None:
            self.pos_encoding = nn.Parameter(data=self.pos_encoding, requires_grad=False)

    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """
        The forward function takes in tensors of token ids and transforms them into vector embeddings. 
        It then adds the positional encoding to the embeddings, and if configured, performs dropout on the layer!

        Args:
            inputs: Batched Sequence of Token Ids

        Shape:
            - inputs: (batch_size, sequence_length)
            - outputs: (batch_size, sequence_length, d_model)
        """

        outputs = None

        # Use fancy indexing to extract the positional encodings until position sequence_length
        sequence_length = inputs.shape[-1]
        pos_encoding = 0
        if self.pos_encoding is not None:
            pos_encoding = self.pos_encoding[:sequence_length]

        ########################################################################
        # TODO:                                                                #
        #   Task 1: Compute the outputs of the embedding layer                 #
        #   Task 4: Add the positional encoding to the output                  #
        #                                                                      #
        # Hint 4: We have already extracted them for you, all you have to do   #
        #         is add them to the embeddings!                               #
        ########################################################################


        if self.pos_encoding is not None and sequence_length > self.pos_encoding.shape[0]:
            raise ValueError("Input sequence exceeds the configured maximum length")
        outputs = self.embedding(inputs) + pos_encoding

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs

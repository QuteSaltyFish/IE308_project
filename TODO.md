# To Do list 
- the preprocess of the data
    - the last two training data should be cutted 
    - the data should be of the same size, which I set is 256x256 (but can be changed in the config.json)   

- The network architecture to be decided.
- The denoise and deblur method to be decided.

# Done list
- Finish dataloader and test it.
- Denoise Net tested, but not good
- add Sobel Module to enable the network to learn from the edges.

# Test list
- use unet to detect the watermark
- use corrupted pic with random line to train the network to identify the line.
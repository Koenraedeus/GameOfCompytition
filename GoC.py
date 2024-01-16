import numpy as np
from itertools import product
from PIL import Image

dead = 0
live = 1

rock = 0
paper = 1
scissor = 2

def ConvolveNeighour(array1, array2=None, Verbose=False):
    '''
    This function takes roughly 6 us
    '''
    if Verbose:
        print(f'{array1.shape=}\t{array2.shape=}')
    
    
    sum = np.sum(array1)
    sum -= array1[1,1]
    return sum           

def DaRules(samplespace):
    '''
    This function takes roughly 15 us
    '''
    
    kernel = np.ones([3,3]).astype('uint8')
    test = ConvolveNeighour(samplespace)
    state = samplespace[1,1]
    
    if state == dead:
        if test == 3:
            # print(f'Rule 4\ttest={test}\tstate={state}')
            return 1          
    
    elif state == live:
        if test < 2:
            # print(f'Rule 1\ttest={test}\tstate={state}')
            return 0
        if  test == 2 or test == 3:
            # print(f'Rule 2\ttest={test}\tstate={state}')
            return 1
        if  test > 3:
            # print(f'Rule 3\ttest={test}\tstate={state}')
            return 0    
       
    # print(f'Rule 5\ttest={test}\tstate={state}') 
    return 0   


def evolution(space):
    
    next_gen = np.zeros(space.shape).astype('uint8')
    
    for (j, i) in product(range(1,space.shape[1]-1), range(1,space.shape[0]-1)):
        sample = np.copy(space[i-1:i+2, j-1:j+2])
        next_gen[i,j] = DaRules(sample)
        
    return next_gen.astype('bool')


def resize(array, factor=1):
    '''
    This functions takes roughly 27 us
    '''
    shape = np.array(array.shape)#.astype('uint8')
    output = np.zeros(shape*factor)#.astype('uint8')
    # print(shape)
    for i,j in product(np.arange(shape[0]), np.arange(shape[1])):
        # print(f'{i = }\t{j=}')
        output[i*factor : (i+1)*factor, j*factor: (j+1)*factor] = array[i,j]
    
    return output.astype('uint8')


def SpacePrint(img):
    
    img = img.astype('uint8')
    img = img * 255
    img = np.clip(0, 255, img)
    if len(img.shape) != 3:
        img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2) 
   
    return img


def init_space(mode='Random', a=0.3, H=5, W=5):
    if mode == 'Osc':
        space = np.zeros([3,3])
        space[:, 1] = 1 
        
    if mode == 'Random':
        space = (np.random.rand(H-2,W-2) + a).astype('uint8')
    edge = np.expand_dims(np.zeros(W-2), axis=0)
    space = np.concatenate([edge, space, edge])
    edge = np.expand_dims(np.zeros(H), axis=0)
    space = np.concatenate([edge, space.T, edge]).T
    
    return space.astype('uint8')#.astype('bool')

def add_icons(img):
    '''
    This functions replaces all pixels with an icond of a rock

    '''
    width, height = img.shape
    output = np.ones([16*width, 16*height, 3]).astype('uint8')*255
    
    icon = np.array(Image.open(r'rock.png')).astype('uint8')
    
    for w, h in product(range(width), range(height)):
        if img[w,h]:
            output[(w*16):(w*16)+16, (h*16):(h*16)+16] = icon
                               
    return output


if __name__ == '__main__':
    print("hoi!")
                               
                        
                               
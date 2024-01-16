import numpy as np
from itertools import product
from PIL import Image

dead = 0
live = 1

irock = 0
ipaper = 1
iscissor = 2

def init_space3d(mode='Random', a=0.05, H=5, W=5):
    if mode == 'Osc':
        space = np.zeros([3,3])
        space[:, 1] = 1 
        
    if mode == 'Random':
        space = (np.random.rand(H-2,W-2, 3) + a).astype('uint8')
    edge = np.expand_dims(np.zeros([W-2,3]), axis=0)
    space = np.concatenate([edge, space, edge])    
    
    edge = np.expand_dims(np.zeros([H, 3]), axis=1)   
    space = np.concatenate([edge, space, edge], axis=1)
        
    return space


def SpacePrint3d(img):
    
    img = img.astype('uint8')
    img = img * 255
    img = np.clip(0, 255, img)
    return img


def evolution3d(space, Verbose=False):
    
    next_gen = np.zeros(space.shape).astype('uint8')
    # print(next_gen.shape)
    
    for (j,i) in product(range(1,space.shape[1]-1), range(1,space.shape[0]-1)):
        sample = np.copy(space[i-1:i+2, j-1:j+2])
        next_gen[i,j] = DaRules3d(sample, Verbose=Verbose)
        
    return next_gen.astype('bool')


def DaRules3d(samplespace, Verbose=False):
    '''
    This function takes roughly 15 us
    '''
    output = np.zeros([1,1,3])    
    
    for i in range(3):
        test = np.sum(samplespace[:,:,i]) - np.sum(samplespace[1,1,i])
        state = samplespace[1,1,i]  
        
        if state == dead:
            if test == 3:
                if Verbose:
                    print(f'Dead {test=}')

                # print(f'Rule 4\ttest={test}\tstate={state}')
                output[0,0,i] = 1          

        elif state == live:
            if Verbose:
                print(f'Alive {test=}')
            if test < 2:
                # print(f'Rule 1\ttest={test}\tstate={state}')
                output[0,0,i] = 0
            if  test == 2 or test == 3:
                # print(f'Rule 2\ttest={test}\tstate={state}')
                output[0,0,i] = 1
            if  test > 3:
                # print(f'Rule 3\ttest={test}\tstate={state}')
                output[0,0,i] = 0    
       
    total = np.sum(output)
    
    if total <= 1:
        return output
    elif total == 3:
        print('lol!')
        return output * 0
    
    # print(output)
    if output[0,0,irock] == 0:
        output[0,0,ipaper] = 0
    elif (output[0,0,ipaper] == 0):
        output[0,0,iscissor] = 0
    else:
        output[0,0,irock] = 0
    return output


def resize3d(array, factor=1):
    '''
    This functions takes roughly 27 us
    '''
    shape = np.array(array.shape)#.astype('uint8')
    newshape = np.array(shape) * [factor, factor, 1]
    output = np.zeros(newshape)#.astype('uint8')
    for i,j in product(np.arange(shape[0]), np.arange(shape[1])):
        # print(f'{i = }\t{j=}')
        output[i*factor : (i+1)*factor, j*factor: (j+1)*factor] = array[i,j]
    
    return output.astype('uint8')


def add_icons3D(img):
    '''
    This functions replaces all pixels with an icond of a rock

    '''
    width, height, _ = img.shape
    output = np.ones([16*width, 16*height, 3]).astype('uint8')*255
    
    rock = np.array(Image.open(r'rock.png')).astype('uint8')
    paper = np.array(Image.open(r'paper.png')).astype('uint8')
    scissors = np.array(Image.open(r'sciccors.png')).astype('uint8')
    
    for w, h in product(range(width), range(height)):
        if sum(img[w,h]) == 0:
            continue
            
        if img[w,h,irock]:
            tmp = np.copy(rock)
        elif img[w,h,ipaper]:
            tmp = np.copy(paper)
        elif img[w,h, iscissor]:
            tmp = np.copy(scissors)
            
        output[w*16:w*16+16, h*16:h*16+16] = tmp
                               
    return output
# Abhishek Bodas Final Q2
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import math

dmax = 6
f = 500
bl = 0.20 #bl

const = 10


def load_files_from_zip():
    image_set = []
    files = sorted(glob.glob("./images/*.png"))
    for file in files:
        image = cv2.imread(file)
        if "left" in file:
            left_image = image
        elif "right" in file:
            right_image = image
    return left_image, right_image



def plot_image(title, image):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.show()


def print_rows(matrix):
    print("Printing 3 rows!")
    print("Row 9: ",matrix[8])
    print("Row 10: ",matrix[9])
    print("Row 11: ",matrix[10])


def rho_value_compare(elem1, m1_mean, elem2, m2_mean):
    if elem1 < m1_mean and elem2 < m2_mean:
        return 0
    elif elem1 > m1_mean and elem2 > m2_mean:
        return 0
    else:
        return 1

#Zcen

def ezcen(matrix1, matrix2, l, k):
    m1_mean = np.mean(matrix1)
    m2_mean = np.mean(matrix2)
    
    cost = 0
    for i in range(0,l):
        for j in range(0,k):
            cost = cost + rho_value_compare(matrix1[i][j], m1_mean, matrix2[i][j], m2_mean)
        
    return cost



def get_matrix(image,i,j):
    test = np.zeros((3,3))
    arr = image.copy()
    i_max = arr.shape[0]-1
    j_max = arr.shape[1]-1

    if i==0 and j==0:
        sub_matrix = arr[i:i+2, j:j+2]
#         print("i0j0:",sub_matrix)
        indxs = np.ix_([1,2], [1,2])
        test[indxs] = sub_matrix

    elif i==i_max and j==i_max:
        sub_matrix = arr[i-1:i+1, j-1:j+1]
#         print("imaxjmax:",sub_matrix)
        indxs = np.ix_([0,1], [0,1])
        test[indxs] = sub_matrix

    elif i==0 and j==j_max:
        sub_matrix = arr[i:i+2, j-1:j+1]
        indxs = np.ix_([1,2], [0,1])
        test[indxs] = sub_matrix

    elif i==i_max and j==0:
        sub_matrix = arr[i-1:i+1, j:j+2]
        indxs = np.ix_([0,1], [1,2])
        test[indxs] = sub_matrix

    elif i==0:
        sub_matrix = arr[i:i+2, j-1:j+2]
        indxs = np.ix_([1,2], [0,1,2])
        test[indxs] = sub_matrix

    elif i==i_max:
        sub_matrix = arr[i-1:i+1, j-1:j+2]
        indxs = np.ix_([0,1], [0,1,2])
        test[indxs] = sub_matrix

    elif j==0:
        sub_matrix = arr[i-1:i+2, j:j+2]
        indxs = np.ix_([0,1,2], [1,2])
        test[indxs] = sub_matrix

    elif j==j_max:
        sub_matrix = arr[i-1:i+2, j-1:j+1]
        indxs = np.ix_([0,1,2], [0,1])
        test[indxs] = sub_matrix

    else:    
        test = arr[i-1:i+2, j-1:j+2]
        
    return test



def fill_msg_board(msg_board, image1, image2, l, k, d):
    matrix1 = np.zeros((l,k)) 
    matrix2 = np.zeros((l,k)) 
    
    
    for i in range(0,len(image1)):
        for j in range(0,len(image1[0])):
            matrix1 = get_matrix(image1, i, j)
            if j-d < 0:
                msg_board[i][j] = 0
                continue
            else:
                matrix2 = get_matrix(image2, i, j-d)

            
            msg_board[i][j] = ezcen(matrix1, matrix2, l, k)
            
            
            
#     for i in range(1,len(image1)-1):
#         for j in range(1,len(image1[0])-1):
#             matrix1 = get_matrix(image1, i, j)
#             if j-d < 0:
#                 msg_board[i][j] = 0
#                 continue
#             else:
#                 matrix2 = get_matrix(image2, i, j-d)

            
#             msg_board[i][j] = ezcen(matrix1, matrix2, l, k)
            
    return msg_board


def initialize_boards(image1, image2, dmax):
    msg_boards = np.zeros((dmax+1, len(image1), len(image1[0])))
    
    for d in range(0,dmax+1):

        msg_boards[d] = fill_msg_board(msg_boards[d], image1, image2, 3, 3, d)
        print("\nMessage Board: ",d, "\t shape: ",msg_boards[d].shape)
    
        print_rows(msg_boards[d])
    return msg_boards



def fill_disparity_matrix(disparity_matrix, msg_boards):
    
    for i in range(0,disparity_matrix.shape[0]):
        for j in range(0,disparity_matrix.shape[1]):
            min_val = 100000000
            min_idx = np.nan
            min_val_counter = 0
            for d in range(0,len(msg_boards)):
                if msg_boards[d][i][j] < min_val:
                    min_val_counter = 0
                    min_idx = d
                    min_val = msg_boards[d][i][j]
                elif msg_boards[d][i][j] == min_val:
                    min_val_counter = min_val_counter + 1

            if min_val_counter > 0:
                min_idx = np.nan
            disparity_matrix[i][j] = min_idx
            
    print("\nDisparity Matrix: ")
    print_rows(disparity_matrix)
            
    return disparity_matrix


def depth(disparity):
    if math.isnan(disparity):
        return 0

    else:
        value = np.divide(np.multiply(f,bl),(disparity+1))
        return np.round(value,4)



def calculate_depth(disparity_matrix):
    depth_matrix = np.zeros(disparity_matrix.shape)
    
    for i in range(0, disparity_matrix.shape[0]):
        for j in range(0, disparity_matrix.shape[1]):
            depth_matrix[i][j] = depth(disparity_matrix[i][j])
            
    print("\nDepth Matrix: ")
    print_rows(depth_matrix)
            
    return depth_matrix


#Belief propagation


def calc_Edata(msg_boards, d, i, j):
    return msg_boards[d][i][j]


def calc_smoothness(x):
    if x==0:
        return 0
    else:
        return const



def print_msg_boards(msg_boards):
    for d in range(0,dmax+1):
        print("\nMessage Board: ",d, "\t shape: ",msg_boards[d].shape)
        print_rows(msg_boards[d])
    return msg_boards



def message_update(pi,pj,qi,qj, t, d, msg_boards):
    max_i = msg_boards[0].shape[0]
    max_j = msg_boards[0].shape[1]
    
    if t == 0:
        return 0
    
    elif pi<0 or pi>=max_i or pj<0 or pj>=max_j or qi<0 or qi>=max_i or qj<0 or qj>=max_j:
        return 0
    
    neighbour1, neighbour2, neighbour3, neighbour4 = 0, 0, 0, 0
    min_message = 100000
    
    for h in range(0, dmax+1):
        msg = calc_Edata(msg_boards, h, pi, pj) + calc_smoothness(h-d) 
        
        if pi-1 != qi and pj != qj:
            neighbour1 = message_update(pi-1,pj,pi,pj, t-1, h, msg_boards)
        if pi+1 != qi and pj != qj:
            neighbour2 = message_update(pi+1,pj,pi,pj, t-1, h, msg_boards)
        if pi != qi and pj-1 != qj:
            neighbour3 = message_update(pi,pj-1,pi,pj, t-1, h, msg_boards)
        if pi != qi and pj+1 != qj:
            neighbour4 = message_update(pi,pj+1,pi,pj, t-1, h, msg_boards)
            
        neighbour_sum = neighbour1 + neighbour2 + neighbour3 + neighbour4
        
        msg = msg + neighbour_sum
        
        if msg < min_message:
            min_message = msg
        
    return min_message



def bpm(msg_boards):
    
    for t in range(0,4):
        print("\n\nIteration :",t)
    
        n1,n2,n3,n4 = 0,0,0,0

        new_msg_board = np.zeros(msg_boards.shape)

        for i in range(0, msg_boards.shape[1]):
            for j in range(0, msg_boards.shape[2]):
                total_cost = 0
                for d in range(0, dmax+1):
                    msg = calc_Edata(msg_boards, d, i, j) 
                    n1 = message_update(i-1, j, i, j, t, d, msg_boards)
                    n2 = message_update(i+1, j, i, j, t, d, msg_boards)
                    n3 = message_update(i, j-1, i, j, t, d, msg_boards)
                    n4 = message_update(i, j+1, i, j, t, d, msg_boards)

                    total_cost = msg + n1 + n2 + n3 + n4

                    new_msg_board[d][i][j] = total_cost

        msg_boards = new_msg_board
        
    return msg_boards
    

def main():
    left_image, right_image = load_files_from_zip()

    
    base_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    match_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    
    plot_image("left", base_image)
    plot_image("right", match_image)
    
    
    msg_boards = initialize_boards(base_image, match_image, 6)
    
    disparity_matrix = np.zeros(base_image.shape)
    
    disparity_matrix = fill_disparity_matrix(disparity_matrix, msg_boards)
    
    plot_image("disparity_matrix", disparity_matrix)
    
    depth_matrix = calculate_depth(disparity_matrix)
    
    depth_image = cv2.normalize(depth_matrix, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    
    plot_image("depth_image", depth_image)
    
    
  
    
    print("msg board shape -------",msg_boards[0].shape)
    msg_boards = bpm(msg_boards)
    
    new_disparity_matrix = fill_disparity_matrix(disparity_matrix, msg_boards)
    plot_image("disparity_matrix", new_disparity_matrix)
    
    new_depth_matrix = calculate_depth(new_disparity_matrix)
    
    new_depth_image = cv2.normalize(new_depth_matrix, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    
    plot_image("left", base_image)
    plot_image("new_depth_image", new_depth_image)



main()





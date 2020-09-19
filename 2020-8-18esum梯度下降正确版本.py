def k_X_dots(k , allpoint):
    # 计算系数相乘点之后的结果，单独的列求和 ， 返回值为（27，1）

    # kshape = 40 ,1
    rows ,cols =k.shape

    # 40 ,27
    allpoint_rows ,allpoint_cols= allpoint.shape
    all_point_new = np.zeros((allpoint_rows,allpoint_cols))
    for i in range(rows):
        for col in range(allpoint_cols):
            all_point_new[i,col] = allpoint[i,col] * k[i,0]
    result = np.zeros((1 , allpoint_cols))

    for row in range(allpoint_rows):
        result[:,:] =result[:,:]+ all_point_new[row,:]
    result =np.resize(result,(allpoint_cols,1))

    # 得到结果(27,1)
    return result
# 创建一个txt文件，文件名为mytxtfile
def text_create(name):
    # 新创建的txt文件的存放路径
    full_path = name   # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')

def makedir(dir):
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False
# 计算原数值与目标值的距离大小
# 原array1与array2尺寸为 （27，1）
def distanceFrom_A_To_B(array1 ,array2 ):
    # 加一个调整系数 ，中间为1 的数值很重要
    rows , cols = array1.shape
    sum =0.0
    for row in range(rows):
        temp_sum = array1[row ,0 ] - array2[row ,0 ]
        sum = sum + temp_sum**2
    sum = sum *1.0
    return  sum
'''
关于梯度下降算法的直观理解，我们以一个人下山为例。比如刚开始的初始位置是在红色的山顶位置，
那么现在的问题是该如何达到蓝色的山底呢？按照梯度下降算法的思想，它将按如下操作达到最低点：

第一步，明确自己现在所处的位置

第二步，找到相对于该位置而言下降最快的方向

第三步， 沿着第二步找到的方向走一小步，到达一个新的位置，此时的位置肯定比原来低

第四部， 回到第一步

第五步，终止于最低点
'''
if __name__ == '__main__':
    # 貌似做小dist 为0.28？？
    # 迭代后的系数保存位置， ----------每次要修改---------

    # 梯度下降对每个参数都要更新，而非更新一个参数

    k_to_save_dir = 'k_to_save//总场强esum非分量计算长的强度设置为100//时间2020-08-16//'
    makedir(k_to_save_dir)



    # 初始化变量f ，全局变量
    eplison = 1e-4  # 小于该误差停止迭代
    max_iter = 1e10  # 大于该循环次数停止迭代
    learning_rate = 0.001# 每次某一个系数改变的大小，方向正或者负 .001比较好
    # 或者learningrate改为迭代次数改变的结果
    learning_rate_dict ={ 'time1' : 0.001 , 'time2' : 0.0001,'time3':0.00001}
    # 中心点强度
    zhongxingdianqiangdu = 1.0 * 100

    for i in range(1):

        import numpy as np
        bochang = 800
        mode_40 = np.load('mode_numpy//40modesum.npy')
        print(mode_40.shape)
        rows, cols = mode_40.shape  # 40 * 801**2尺寸
        mode_number = min( rows ,cols)
        jingdu =int( ( max(rows ,cols) ) **0.5)

        mode_80 = np.load('mode_numpy//80mode.npy')
        # rows_80 801**2 , cols_80 480
        rows_80 ,cols_80 = mode_80.shape


        # 26.433
        banfenkuang = bochang * 0.33 / 10

        # 801//2
        middle_point = jingdu //2

        # 26
        banfenkuang_int = int(banfenkuang)

        # 设置的点的个数，一个中心点， 26个零点, 共27个点
        numberofPreset = 2  *banfenkuang_int - banfenkuang_int +1

        zuobiao = [ (middle_point,middle_point) ]
        result_wanted = [zhongxingdianqiangdu]
        # -----------------------------保存初始的随机系数--------------
        k=np.random.randn(mode_number,1)
        saveplace = k_to_save_dir + 'No0' + '.npy'
        np.save(saveplace, k)


        for i in range( banfenkuang_int , 2*banfenkuang_int , 1) :
            result_wanted.append(0.0)
            zuobiao_x  = middle_point
            zuobiao_y =  middle_point + i
            zuobiao.append( (zuobiao_x , zuobiao_y ) )
    print(zuobiao)
    #想要的结果是（27，1）
    wanted_point =np.resize( np.array(result_wanted) , (numberofPreset, 1 ))

    # 40个模式，每个提取27个预设点 尺寸为（40*27 ）
    all_point = np.zeros((mode_number , numberofPreset))

    for number in range(mode_number):
        temp_mode = np.resize( mode_40[number , :]  ,new_shape=(jingdu,jingdu) )

        # 第几个点 ,
        numberPoint = 0
        for  x, y in zuobiao:
            # 27 个坐标
            x= int(x)
            y= int(y)
            all_point[number ,numberPoint ]= temp_mode[x, y]
            numberPoint = numberPoint +1
    all_point =all_point # (40,27)尺寸
    old_point = k_X_dots(k=k,allpoint=all_point)
    dist = distanceFrom_A_To_B(wanted_point , old_point)
    print(dist)
    # 循环次数count

    count = 0
    # 保留副本，原系数1/40 ， 1/40 .。。。
    k_old  =k.copy()
    # k[-1,0] =-.795
    # k[-2,0] = -0.095

    dist_min = dist
    while dist>eplison: # 误差小于预设误差，退出迭代
        dist_old = dist
        count = count + 1
        if count>max_iter:break # 大于最大循环次数 ， 退出迭代
        k_to_change = 0
        sign_to_change = 1

        dist_min = dist_old
        for temp in range(mode_number):
            k_new_postive= k.copy()
            k_new_negative = k.copy()
            # 正向变化的值,
            k_new_postive[temp,0] =k_new_postive[temp,0] + learning_rate * 1
            k_new_negative[temp, 0] = k_new_negative[temp, 0] + learning_rate * (-1)
            dist_new_postive = distanceFrom_A_To_B( k_X_dots(k_new_postive , all_point), wanted_point)
            dist_new_negative = distanceFrom_A_To_B( k_X_dots(k_new_negative , all_point), wanted_point)

            # 假使新改变的某一个系数，其带来的结果使得更靠近想要的结果 ，
            if (dist_new_postive < dist_min or dist_new_negative< dist_min):
                dist_min = min(dist_new_postive, dist_new_negative)
                if (dist_new_postive  < dist_new_negative):
                    sign_to_change =1
                    k_to_change = temp
                else:
                    sign_to_change = -1
                    k_to_change = temp


        if (sign_to_change == 1):
            print('迭代次数是', count, ',要改变的模式系数是第', k_to_change + 1, '个', '当前的dist是', dist_min, '要改变的符号方向是 + ',learning_rate  )
        if (sign_to_change == -1):
            print('迭代次数是', count, ',要改变的模式系数是第', k_to_change + 1, '个', '当前的dist是', dist_min, '要改变的符号方向是 - '  ,learning_rate)
        # print(k)
        # 计算出来的系数更新 ，以及距离更新
        dist = dist_min
        k[k_to_change , 0] = k[k_to_change , 0] + learning_rate * sign_to_change
        # 每1000次保存一次数据
        if(count %1000 ==0):
            print(k)
            saveplace = k_to_save_dir+'No'+ str(count)+'.npy'
            np.save(saveplace , k)


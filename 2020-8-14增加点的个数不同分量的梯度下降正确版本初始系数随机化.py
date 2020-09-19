def k_X_dots(k , allpoint):
    #k 是40，1 ， all point 是 40，27

       # 计算系数相乘点之后的结果，单独的列求和 ， 返回值为（27，1）

    # kshape = 40 ,1
    rows ,cols =k.shape

    # 40 ,27
    allpoint_rows ,allpoint_cols= allpoint.shape
    all_point_new = np.zeros((allpoint_rows,allpoint_cols),dtype=complex)
    for i in range(rows):
        for col in range(allpoint_cols):
            all_point_new[i,col] = allpoint[i,col] * k[i,0]
    result = np.zeros((1 , allpoint_cols),dtype=complex)

    for row in range(allpoint_rows):
        result[0,:] =result[0,:]+ all_point_new[row,:]
    result =np.resize(result,(allpoint_cols,1))

    # 得到结果(27,1)
    return result


def k3_40_1_X40_3_27_dots(k, allpoint):
    # 返回结果是3，27

    # k 是3，40，1 ，
    # all point 是 40，3，26*2+1
    allpoint_a,allpoint_b, allpoint_c = allpoint.shape

    # 计算系数相乘点之后的结果，单独的列求和 ， 返回值为（27，1）
    a,b,c = k.shape
    allpoint_result = np.zeros(allpoint.shape ,dtype=complex )
    modenumber = allpoint_a
    weidu =min(allpoint_a ,allpoint_b,allpoint_c)

    for mode in range(modenumber):
        for exeyezhere in range(weidu):
            allpoint_result[mode,exeyezhere,:] = k[exeyezhere,mode ,0] * allpoint[mode ,exeyezhere , :]
    # 3 ,27 维度的结果 ，3 代表各自的分量exeyez ，27个点
    finally_result = np.zeros((allpoint_b,allpoint_c),dtype=complex)

    for dian_number in range(allpoint_c):
        for fengliangexeyez in range(allpoint_b):
            for mode in range(modenumber):
                finally_result[fengliangexeyez,dian_number] += (allpoint_result[mode,fengliangexeyez,dian_number] )
    #  # 返回结果是3，26*2+1
    return finally_result


# 创建一个txt文件，文件名为mytxtfile
def text_create(name):
    # 新创建的txt文件的存放路径
    full_path = name   # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')


def trans_3_27_to_esum(all_point):
    # 返回结果是27，1 ，
    # 输入是3，27
    import numpy as np
    all_point = np.array(all_point)
    # a 3   b 27
    a,b= all_point.shape

    mode_temp = np.zeros((1, b) )

    for j in range(b):
        for i in range(a):
            mode_temp[0  , j ] += np.abs(all_point [i ,j])

    mode_result = np.resize(mode_temp,new_shape=(b,1))
    # 返回值为27，1.实数矩阵
    return  mode_result

# 40 ,3 ,27 复数矩阵  转换为 27 ，1 的实数矩阵 , 我觉得转换为3 ， 27 维度更好
def trans_exeyez_to_esum(all_point):
    all_point = np.array(all_point)
    # a 40  b 3 c 27
    a,b,c = all_point.shape

    mode_temp = np.zeros((b, c) ,dtype=complex)

    for i in range(b):
        for point in range(c):
            for mode in range(a):
                mode_temp[i  ,point] += np.abs(all_point[mode , i , point])
    mode_result = np.zeros((c,1))
    for point in range(c):
        for i in range(b):
            mode_result[c,0]  += np.abs( mode_temp[i,c] )
    mode_result = np.resize(mode_result,new_shape=(c,1))
    return  mode_result



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
    fengliangtochange={0:'ex' ,  1 :'ey',2:'ez'}
    shibuxubu ={0:'实数部',  1:'虚数部'}
    # 貌似最小dist 为0.28？？就死循环了
    import numpy as np


    # -------------------------------------迭代后的系数保存位置，多少次迭代保存一次更新的系数矩阵 ----------每次要修改---------
    k_to_save_dir = r'k_to_save//分量各自迭代保存的系数结果//2020-8-22//'
    update_time = 500
    # -------------------------------------迭代后的系数保存位置，多少次迭代保存一次更新的系数矩阵 ----------每次要修改---------
    makedir(k_to_save_dir)
    # 初始化变量f ，全局变量
    eplison =0.00001 # 小于该误差停止迭代
    max_iter = 1e10  # 大于该循环次数停止迭代
    learning_rate = 0.001  # 每次某一个系数改变的大小，方向正或者负 , 学习率

    # 中心点强度
    zhongxingdianqiangdu = 1.0 * 10
    # 读取原始数据mode
    mode_org = r'mode_numpy//80mode.npy'

    # 原始数据
    mode_orign = np.load(mode_org)
    a, b = mode_orign.shape
    jingdu = int((max(a, b)) ** 0.5) # 801精度

    mode_number = 40  # 40个模式
    mode_fenliang = 3 # ex ey ez分量

    # 波长
    bochang = 800
    # 26.433
    banfenkuang = bochang * 0.33 / 10
    # 801//2
    middle_point = jingdu // 2
    # 26
    banfenkuang_int = int(banfenkuang)
    # 设置的点的个数，一个中心点， 26个零点, 共27个点
    numberofPreset = 3 * banfenkuang_int - banfenkuang_int + 1 # 27

    zuobiao = [(middle_point, middle_point)]
    result_wanted = [zhongxingdianqiangdu] # 中心点强度设置

    # 预设结果初始化，想要得到的结果： （27，1）结果   （坐标为）（27个点）
    for i in range( banfenkuang_int , 3 * banfenkuang_int , 1) :
        result_wanted.append(0.0)
        zuobiao_x  = middle_point
        zuobiao_y =  middle_point + i
        zuobiao.append( ( zuobiao_x , zuobiao_y ) )
    print(zuobiao)
    # wanted point  (27,1)
    wanted_point = np.resize(np.array(result_wanted), (numberofPreset, 1))

    #得到复数矩阵
    mode_fushu = np.zeros((mode_number, mode_fenliang, jingdu * jingdu), dtype=complex)  # 40 , 3 ,801**2
    # 得到复数矩阵 ， 【40，3，801*801】维度 , 0维度是ex ，1维度是ey ，2 维度是ez
    for col in range(jingdu ** 2):
        for i in range(mode_number):
            ex_start = i * 12 + 0
            ey_start = i * 12 + 2
            ez_start = i * 12 + 4
            mode_fushu[i, 0, col] += mode_orign[col, ex_start] + 1j * mode_orign[col, ex_start + 1]
            mode_fushu[i, 1, col] += mode_orign[col, ey_start] + 1j * mode_orign[col, ey_start + 1]
            mode_fushu[i, 2, col] += mode_orign[col, ez_start] + 1j * mode_orign[col, ez_start + 1]
    # mode_fushu 是40个本征模，三个维度，且
    mode_fushu  = mode_fushu


    # 40个模式，每个提取3维度  ， 27个预设点 尺寸为（40 ， 3 ， 27 ） # 各自分量
    all_point = np.zeros((mode_number,mode_fenliang, numberofPreset), dtype=complex)
    for number in range(mode_number):
        for fenliang in range(mode_fenliang):
            temp_mode = np.resize(mode_fushu[number, fenliang ,:], new_shape=(jingdu, jingdu))
            # 第几个点 ,
            numberPoint = 0
            for x, y in zuobiao:
                # 27 个坐标
                x = int(x)
                y = int(y)
                all_point[number,fenliang ,numberPoint] = temp_mode[x, y]
                numberPoint = numberPoint + 1
    # 得到allpoint位置 （40 ，3 ，27） 3代表ex ，ey，ez
    # 系数矩阵初始化 ，都是0矩阵 复数矩阵 ， （3，40，1）规格
    k_ex_ey_ez = np.zeros((mode_fenliang, mode_number, 1), dtype=complex)

    # 保存初始化矩阵
    saveplace = k_to_save_dir + 'No' + str(0) + '.npy'
    np.save(saveplace, k_ex_ey_ez)

    # ------------------------------初始化矩阵也可以改用随机化的初始化矩阵------------------------------------
    # 好像随机化更好一点！！！！！！！！！！！！！！！！！！！！！！！！@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # # 初始化三个系数矩阵 ， 但是感觉问题不大，没什么用了
    # for chushihuafenliang in range(1):
    #     # 初始化三个分量ex ey ez 都是（mode_number ， 1）维度的复数矩阵
    #     for ex in range(1):
    #         kreal = np.random.randn(mode_number,1)
    #         kimg =np.random.randn(mode_number,1)
    #         k_ex = np.zeros((mode_number,1) ,dtype=complex)
    #         for num in range(mode_number):
    #             k_ex[num , 0 ] =  kreal[num , 0 ] + 1j * kimg[num, 0]
    #     for ey in range(1):
    #         kreal = np.random.randn(mode_number, 1)
    #         kimg = np.random.randn(mode_number, 1)
    #         k_ey = np.zeros((mode_number, 1), dtype=complex)
    #         for num in range(mode_number):
    #             k_ey[num, 0] = kreal[num, 0] + 1j * kimg[num, 0]
    #     for ez in range(1):
    #         kreal = np.random.randn(mode_number, 1)
    #         kimg = np.random.randn(mode_number, 1)
    #         k_ez = np.zeros((mode_number, 1), dtype=complex)
    #         for num in range(mode_number):
    #             k_ez[num, 0] = kreal[num, 0] + 1j * kimg[num, 0]
    # # 得到结果是3个 （ mode_number， 1） 维度的复数矩阵k_ex , k_ey , k_ez, 保存迭代前的数值信息
    # print(k_ex, k_ey , k_ez)

    # # 将初始化的系数保存到一个矩阵里面
    # for mode in range(mode_number):
    #     k_ex_ey_ez[0, mode, 0] = k_ex[mode, 0]
    #     k_ex_ey_ez[1, mode, 0] = k_ey[mode, 0]
    #     k_ex_ey_ez[2, mode, 0] = k_ez[mode, 0]
    # # 维度 3，40，1
    # k_ex_ey_ez =k_ex_ey_ez
    #
    # # 保存初始的随机系数
    # for baocun in range(1):
    #     saveplace = k_to_save_dir + 'No0' + 'ex.npy'
    #     np.save(saveplace, k_ex)
    #     saveplace = k_to_save_dir + 'No0' + 'ey.npy'
    #     np.save(saveplace, k_ey)
    #     saveplace = k_to_save_dir + 'No0' + 'ez.npy'
    #     np.save(saveplace, k_ez)
    #     saveplace = k_to_save_dir + 'No0' + 'exeyez.npy'
    #     np.save(saveplace, k_ex_ey_ez)
    #
    # # 对待各个分量计算系数相乘模式,得到27，1
    # ex_new = k_X_dots(k_ex , np.resize(all_point[:,0,:] , new_shape=(mode_number, numberofPreset) ) )
    # ey_new = k_X_dots(k_ey , np.resize(all_point[:,1,:] , new_shape=(mode_number, numberofPreset) ) )
    # ez_new = k_X_dots(k_ez , np.resize(all_point[:,2,:] , new_shape=(mode_number, numberofPreset) ) )
    # all_point_temp =np.zeros((3,numberofPreset),dtype=complex)
    #
    # for i in range(numberofPreset):
    #     all_point_temp[0,i] = ex_new[i,0] ;
    #     all_point_temp[1,i] =  ey_new[i,0] ;
    #     all_point_temp[2,i] =  ez_new[i,0] ;
    # # 得到27，1 维度的矩阵
    # esum_new = trans_3_27_to_esum(all_point_temp)
    # # 原始距离
    #
    # dist_origin1  = distanceFrom_A_To_B(esum_new , wanted_point)
    # dist = dist_origin1 *1.0


    # 证明esum2 和esun1计算相同,但是不相同，可是更靠谱的应该是下面那个


    esum_new2 = trans_3_27_to_esum(
        k3_40_1_X40_3_27_dots(k_ex_ey_ez, all_point )
    )
    dist_origin2 =distanceFrom_A_To_B(esum_new2 ,wanted_point)
    dist2 = dist_origin2 * 1.0

    count = 0 # # 计数器

    dist_min = dist2

    # 开始迭代循环
    while dist_min > eplison:  # 误差小于预设误差，退出迭代
        dis_old =dist_min #
        count = count + 1
        if count > max_iter: break  # 大于最大循环次数 ， 退出遍历
        k_to_change = 0 # 第几个模式的系数要改变
        real_or_img = 0 # 0代表要改变的系数是实部 ， 1  代表的是虚部
        k_to_change_ex_or_ey_or_ez = 0 # 0代表要改变的是ex的系数 ， 1 代表ey ，2代表ez
        sign_to_change = 1 # 1代表正向改变 ， -1代表负向改变

        for mode in range(mode_number): # 哪一个模式的系数改变
            for exeyez in range(mode_fenliang): # 该模式的哪一个分量改变？（）
                for realorimg in range(2):        # 该分量是实数部分还是虚数部分改变
                    if(realorimg == 0): #实数部分改变
                        k_new_postive = k_ex_ey_ez.copy()
                        k_new_negative = k_ex_ey_ez.copy()
                        # -----------------正向变化的值,和负向变化的值比较----------
                        k_new_postive[exeyez,mode, 0] = k_new_postive[exeyez,mode, 0] + learning_rate * 1
                        k_new_negative[exeyez,mode, 0] = k_new_negative[exeyez,mode, 0]  + learning_rate * (-1)
                        dist_new_postive = distanceFrom_A_To_B(
                            trans_3_27_to_esum(
                                k3_40_1_X40_3_27_dots(k_new_postive, all_point)  ),
                            wanted_point
                        )
                        dist_new_negative  = distanceFrom_A_To_B(
                            trans_3_27_to_esum(
                                k3_40_1_X40_3_27_dots(k_new_negative, all_point)),
                            wanted_point
                        )
                        # 假使实数部分新改变的某一个系数，其带来的结果使得更靠近想要的结果distmin ，
                        if (dist_new_postive < dist_min or dist_new_negative < dist_min):
                            dist_min = min(dist_new_postive, dist_new_negative)
                            if (dist_new_postive < dist_new_negative):
                                real_or_img =  0
                                sign_to_change = 1
                                k_to_change = mode
                                k_to_change_ex_or_ey_or_ez = exeyez
                            else:
                                real_or_img = 0
                                sign_to_change = -1
                                k_to_change = mode
                                k_to_change_ex_or_ey_or_ez = exeyez


                    if (realorimg ==1): # 虚数部分改变
                        k_new_postive = k_ex_ey_ez.copy()
                        k_new_negative = k_ex_ey_ez.copy()
                        # 正向变化的值,和负向变化的值比较
                        k_new_postive[exeyez,mode, 0] = k_new_postive[exeyez,mode, 0] + learning_rate * (1)* ( 1j)
                        k_new_negative[exeyez,mode, 0] = k_new_negative[exeyez,mode, 0]  + learning_rate * (-1)* ( 1j)
                        dist_new_postive = distanceFrom_A_To_B(
                            trans_3_27_to_esum(
                                k3_40_1_X40_3_27_dots(k_new_postive, all_point)  ),
                            wanted_point
                        )
                        dist_new_negative  = distanceFrom_A_To_B(
                            trans_3_27_to_esum(
                                k3_40_1_X40_3_27_dots(k_new_negative, all_point)),
                            wanted_point
                        )
                        # 假使新改变的某一个系数，其带来的结果使得更靠近想要的结果 ，
                        if (dist_new_postive < dist_min or dist_new_negative < dist_min):
                            dist_min = min(dist_new_postive, dist_new_negative)
                            if (dist_new_postive < dist_new_negative):
                                real_or_img =  1
                                sign_to_change = 1
                                k_to_change = mode
                                k_to_change_ex_or_ey_or_ez = exeyez
                            else:
                                real_or_img = 1
                                sign_to_change = -1
                                k_to_change = mode
                                k_to_change_ex_or_ey_or_ez = exeyez


        # 如果有dist有减小的话，进行循环  和 系数变更
        if (dist_min < dis_old):
            if (sign_to_change == 1):  #  是正向变化的话
                print(
                    '迭代次数是', count, ' 当前的dist是', dist_min,'要改变的模式系数是第', k_to_change + 1, '个',
                      '要改变的分量是',fengliangtochange[k_to_change_ex_or_ey_or_ez],'要改变的是实部还是虚部',
                      shibuxubu[real_or_img],'要改变的符号方向是 + ',
                      learning_rate
                )
            if (sign_to_change == -1): # 负向变化的话
                print(
                    '迭代次数是', count, ' 当前的dist是', dist_min,'要改变的模式系数是第', k_to_change + 1, '个',
                      '要改变的分量是',fengliangtochange[k_to_change_ex_or_ey_or_ez],'要改变的是实部还是虚部',
                      shibuxubu[real_or_img],'要改变的符号方向是 - ',
                      learning_rate
                )
            # ---------------------------你是猪吗？？？？？？ ？？？？？？？ ？？？？？？？？？？？？？？？？？？？？下面是错的
            # 对系数进行修改,-----实部改变
            if real_or_img == 0:
                k_ex_ey_ez[k_to_change_ex_or_ey_or_ez, k_to_change, 0] = \
                    k_ex_ey_ez[k_to_change_ex_or_ey_or_ez, k_to_change, 0]  +  1 * learning_rate * sign_to_change
            # 对系数进行修改,----虚数部改变
            if real_or_img == 1:
                k_ex_ey_ez[k_to_change_ex_or_ey_or_ez, k_to_change, 0] = \
                    k_ex_ey_ez[k_to_change_ex_or_ey_or_ez, k_to_change, 0] +   1* learning_rate * sign_to_change*(1j)

            # --------------------------你是猪吗？？？？？ ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？上面是错的
            # 当前的最小距离
            dist_min = dist_min

            # 每500次保存一次数据
            if (count % update_time == 0):
                print(k_ex_ey_ez)
                saveplace = k_to_save_dir + 'No' + str(count) + '.npy'
                np.save(saveplace, k_ex_ey_ez)
        # --------------------------    这个好像不重要！先省略
        # else:
        #     print('循环结束迭代结束','循环了count次数 = ' ,count);
        #     break;
    print('初始的dist是',dist2)
    print('一共循环了count次数 = ' ,count,' 当前的dist是', dist_min)


# 查看系数所得的结果图

# 新建文件保存图片
def makedir(dir):
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    import numpy as np
    mode_save = 'mode_numpy//40modesum.npy'
    mode_sum = np.load(mode_save)
    rows, cols = mode_sum.shape
    jingdu = int(cols ** 0.5)
    print(rows, cols)
    # 选择哪个系数文档 ---------------要修改-------------------------------------
    for number in range(0,100000,500):
        No =  int(number)
        k_save ='k_to_save//No' + str(No)+'.npy'

        pic_save_dir = 'pic_to_save//allesum2020-8-22//No' + str(No)
        makedir(pic_save_dir)
        k = np.load(k_save)
        print(k.shape)

        for i in range(rows):
            mode_sum[i,:] = mode_sum[i,:] * k[i]
        mode_result =np.zeros(( 1 , cols))
        for i in range(rows):
            mode_result[0,:] += mode_sum[i,:]
        print(mode_result)

        # 叠加的结果 去掉绝对值
        esum_temp = np.resize(np.abs(mode_result) ,  (jingdu, jingdu))

        plt.ion()
        plt.imshow(esum_temp, cmap='plasma')
        plt.title('ex的第' + '个解，所采用的系数对所有模式叠加得到的强度结果图')
        cbar = plt.colorbar()
        cbar.set_label('intensity', rotation=-90, va='bottom')

        max_index = np.max(esum_temp)
        min_index = np.min(esum_temp)
        interval_temp = (max_index - min_index) / 5
        cbar.set_ticks([min_index, min_index + 1 * interval_temp,
                        min_index + 2 * interval_temp, min_index + 3 * interval_temp,
                        min_index + 4 * interval_temp, min_index + 5 * interval_temp])
        # set the font size of colorbar
        cbar.ax.tick_params(labelsize=8)
        save_place = pic_save_dir + '//' + 'ex' + '.png'
        plt.savefig(save_place)
        plt.pause(0.005)
        plt.close('all')

        # 画出red 和black的图像
        for bilibli in range(1):
            for red in range(1):
                plt.ion()
                # red 保存
                # 寻找沿着y轴中心的画图方式
                x_label = np.arange(-1 * ((jingdu) // 4), (jingdu) // 4 + 1)
                # 归一化吗?no

                nouse_red = np.array(
                    esum_temp[(jingdu // 2 - (jingdu // 4)):(jingdu // 2 + (jingdu // 4) + 1),
                    (jingdu // 2 - (jingdu // 4)): ((jingdu // 2) + (jingdu // 4) + 1)]
                )
                y_label = (
                    np.abs(np.resize(nouse_red[jingdu // 4, :], (jingdu // 2 + 1, 1)))
                )
                pic_save_dir_red = pic_save_dir + '//red'
                makedir(pic_save_dir_red)
                save_place = pic_save_dir_red + '//'  + 'mode_red' + '.png'

                plt.title('red-mode序号是：' )
                plt.plot(x_label, y_label)
                plt.savefig(save_place)
                plt.show()
                plt.pause(0.005)
                plt.close('all')

            # 画出black
            for black in range(1):
                plt.ion()
                # 对黑线进行画图
                x_label = np.arange(-1 * ((jingdu) // 4), (jingdu) // 4 + 1)
                print('xlabel is ', x_label, x_label.shape)
                # 归一化吗? no!
                nouse_black = np.array(
                    esum_temp[jingdu // 2 - jingdu // 4:jingdu // 2 + jingdu // 4 + 1,
                    jingdu // 2 - jingdu // 4:jingdu // 2 + jingdu // 4 + 1]
                )
                y_label = (
                    np.abs(np.resize(nouse_black[:, jingdu // 4], (jingdu // 2 + 1, 1)))
                )
                print('ylabel is ', y_label, y_label.shape)
                pic_save_dir_black = pic_save_dir + '//black'
                makedir(pic_save_dir_black)

                save_place = pic_save_dir_black + '//' + 'mode_black' + '.png'

                plt.title('black-mode的序号是：' )
                plt.plot(x_label, y_label)
                plt.savefig(save_place)
                plt.show()
                plt.pause(0.005)
                plt.close('all')

        # 沿着对角线01画出图像：
        for bibi in range(1):
            # 对角01
            plt.ion()
            # red 保存
            # 寻找沿着对角线轴中心的画图方式，点的个数都是相同的
            x_label = np.arange(-1 * ((jingdu) // 4), (jingdu) // 4 + 1)
            # 归一化吗?no

            duijiaoDict = []
            for temp_number in range(jingdu):
                duijiaoDict.append(esum_temp[temp_number, temp_number])
            nouse_duijiao = np.resize(duijiaoDict, (jingdu, 1))

            y_label = (
                np.abs(np.resize(nouse_duijiao[jingdu // 2 - jingdu // 4:jingdu // 2 + jingdu // 4 + 1],
                                 (jingdu // 2 + 1, 1)))
            )
            pic_save_dir_duijiao = pic_save_dir + '//duijiao01'
            makedir(pic_save_dir_duijiao)
            save_place = pic_save_dir_duijiao + '//' + 'mode_duijiao01' + '.png'

            plt.title('duijiao01-mode序号是：' )
            plt.plot(x_label, y_label)
            plt.savefig(save_place)
            plt.show()
            plt.pause(0.005)
            plt.close('all')

        # 沿着对角线02画出图像：
        for bibi in range(1):
            # 对角01
            plt.ion()
            # red 保存
            # 寻找沿着对角线轴中心的画图方式，点的个数都是相同的
            x_label = np.arange(-1 * ((jingdu) // 4), (jingdu) // 4 + 1)
            # 归一化吗?no

            duijiaoDict = []
            for temp_number in range(jingdu):
                duijiaoDict.append(esum_temp[temp_number, jingdu - temp_number - 1])
            nouse_duijiao = np.resize(duijiaoDict, (jingdu, 1))

            y_label = (
                np.abs(np.resize(nouse_duijiao[jingdu // 2 - jingdu // 4:jingdu // 2 + jingdu // 4 + 1],
                                 (jingdu // 2 + 1, 1))
                       )
            )

            pic_save_dir_duijiao = pic_save_dir + '//duijiao02'
            makedir(pic_save_dir_duijiao)
            save_place = pic_save_dir_duijiao + '//' + 'mode_duijiao02' + '.png'

            plt.title('duijiao02-mode序号是：' )
            plt.plot(x_label, y_label)
            plt.savefig(save_place)
            plt.show()
            plt.pause(0.005)
            plt.close('all')

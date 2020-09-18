import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
算法流程
首先要反转原图像，因为算法之后所有的操作都将0作为前景，将1作为背景。
中心像素x_1(x,y)的8-近邻定义如下所示：
中心像素x_1(x,y)的8-近邻

考虑以下两个步骤

步骤1：执行光栅扫描并标记满足以下5个条件的所有像素：
这是一个黑色像素；
顺时针查看x2、x3、...、x9、x2时，从0到1的变化次数仅为1;
x2、x3、...、x9中1的个数在2个以上6个以下;
x2、x4、x6中至少有1个为1;
x4、x6、x8中至少有1个为1;
将满足条件的所有像素标为1
步骤2：执行光栅扫描并标记满足以下5个条件的所有像素：
这是一个黑色像素；
顺时针查看x2、x3、...、x9、x2时，从0到1的变化次数仅为1;
x2、x3、...、x9中1的个数在2个以上6个以下;
x2、x4、x8中至少有1个为1;
x2、x6、x8中至少有1个为1;
将满足条件的所有像素标为1
反复执行步骤1和步骤2，直到没有点发生变化。
'''
# zhangsuen 细化算法
# Zhang Suen thining algorythm
def Zhang_Suen_thining(img):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # get shape
    H, W, C = img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int)
    out[img[..., 0] > 0] = 1

    # inverse
    out = 1 - out

    while True:
        s1 = []
        s2 = []

        # step 1 ( rasta scan )
        for y in range(1, H - 1):
            for x in range(1, W - 1):

                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1:
                    f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1:
                    f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                # condition 3
                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                # condition 4
                # x2 x4 x6
                if (out[y - 1, x] + out[y, x + 1] + out[y + 1, x]) < 1:
                    continue

                # condition 5
                # x4 x6 x8
                if (out[y, x + 1] + out[y + 1, x] + out[y, x - 1]) < 1:
                    continue

                s1.append([y, x])

        for v in s1:
            out[v[0], v[1]] = 1

        # step 2 ( rasta scan )
        for y in range(1, H - 1):
            for x in range(1, W - 1):

                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y - 1, x + 1] - out[y - 1, x]) == 1:
                    f1 += 1
                if (out[y, x + 1] - out[y - 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x + 1] - out[y, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x] - out[y + 1, x + 1]) == 1:
                    f1 += 1
                if (out[y + 1, x - 1] - out[y + 1, x]) == 1:
                    f1 += 1
                if (out[y, x - 1] - out[y + 1, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x - 1] - out[y, x - 1]) == 1:
                    f1 += 1
                if (out[y - 1, x] - out[y - 1, x - 1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue

                # condition 3
                f2 = np.sum(out[y - 1:y + 2, x - 1:x + 2])
                if f2 < 2 or f2 > 6:
                    continue

                # condition 4
                # x2 x4 x8
                if (out[y - 1, x] + out[y, x + 1] + out[y, x - 1]) < 1:
                    continue

                # condition 5
                # x2 x6 x8
                if (out[y - 1, x] + out[y + 1, x] + out[y, x - 1]) < 1:
                    continue

                s2.append([y, x])

        for v in s2:
            out[v[0], v[1]] = 1

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break

    out = 1 - out
    out = out.astype(np.uint8) * 255

    return out

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
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    pic_origin_dir  = 'F:/Anime-InPainting/data_origin/'
    pic_create_thin_pic_dir = 'F:/Anime-InPainting/data_origin_thin/'
    makedir(pic_create_thin_pic_dir)
    for t in range(100):
        t = str(t);
        t_buquan = t.zfill(4)
        for i in range(100):
            # 展示掩膜图片
            filename =  t_buquan + '_00_' + str(i+1) + '.png'
            # 原图如下
            pic_saveplace = pic_origin_dir + filename
            # Read image
            img = cv2.imread(pic_saveplace).astype(np.float32)

            # Zhang Suen thining
            out = Zhang_Suen_thining(img)
            # 新建的图保存地址
            pic_thin  = pic_create_thin_pic_dir + filename
            # Save result
            cv2.imwrite(pic_thin, out)
            cv2.imshow("result", out)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
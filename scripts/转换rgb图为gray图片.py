def transRGB2GRAY(origin_pic_gray, output_rgb_pic):
    import numpy as np
    import cv2
    file_saveplace = origin_pic_gray
    # the path must be english ,no chinese !
    src = cv2.imread(file_saveplace,0)

    # print(src)
    src_RGB = src
    # print(src_RGB)

    cv2.imwrite(output_rgb_pic, src_RGB)
    cv2.imshow("rgb", src_RGB)
    # 停留1ms
    cv2.waitKey(1)
    cv2.destroyAllWindows()


def makedir(dir):
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False

# 遍历得到某个文件夹下的所有文件名
def getFileName(dirName):
    import os
    fileList= []
    filePath = dirName
    for i, j, k in os.walk(filePath):
        # i是当前路径，j得到文件夹名字，k得到文件名字
        print(i, j, k)
        fileList.append(k)
    return fileList[1]

if __name__ == '__main__':
    # 要转换的图片目录是
    origin_rgb_filePath = 'F://edge-ours//result//yanzheng//'
    file_tochange_filePath = origin_rgb_filePath + 'gray//'
    makedir(file_tochange_filePath )
    filelist = getFileName(origin_rgb_filePath)
    for i in filelist[0]:
        origin_filename = origin_rgb_filePath + i
        rgb_filename = file_tochange_filePath + i
        transRGB2GRAY(origin_filename, rgb_filename)
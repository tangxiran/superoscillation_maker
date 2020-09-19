
import numpy as np


dist= np.load('k_to_save//loss_record.npy')
length  =(dist.size)
import matplotlib.pyplot as plt
x_label = np.arange(length)
y_label =np.resize( dist , new_shape=(length,1)) - 510
print(y_label)
plt.title('loss_iterstion' )
plt.plot(x_label, y_label)
plt.show()
plt.imshow()


# import matplotlib.pyplot as plt
# x_label = np.arange(200000)
#
# plt.title('loss_iterstion' )
# plt.plot(x_label, y_label)
# plt.show()
# plt.imshow()
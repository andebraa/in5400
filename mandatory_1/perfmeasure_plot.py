import numpy as np
import matplotlib.pyplot as plt
#
# perfmeasure = np.load('ssh_copy/perfmeasure.npy')
# print(np.shape(perfmeasure))
#
# perfmeasure2 = np.load('ssh_copy2/perfmeasure.npy')
# print(np.shape(perfmeasure2))
#
# # labels = [
# #     ['aeroplane'], ['bicycle'], ['bird'], ['boat'],
# #     ['bottle'], ['bus'], ['car'], ['cat'], ['chair'],
# #     ['cow'], ['diningtable'], ['dog'], ['horse'],
# #     ['motorbike'], ['person'], ['pottedplant'],
# #     ['sheep'], ['sofa'], ['train'],
# #     ['tvmonitor']]
#
# plt.plot(perfmeasure2.T)
# plt.xlabel('epochs')
# plt.ylabel('average precision per class')
# plt.legend()
# plt.show()
#
# avg_measure = np.average(perfmeasure2, axis=0)
# plt.plot(avg_measure)
# plt.xlabel('epochs')
# plt.ylabel('average precision for all classes')
# plt.show()

trainloss = np.load('ssh_copy2/trainloss.npy')
testloss = np.load('ssh_copy2/testloss.npy')
print(trainloss)
print(testloss)
plt.plot(trainloss, label='train loss')
plt.plot(testloss, label='test loss')
plt.legend()
plt.xlabel('epochs')
plt.show()

import struct
import numpy as np
import  matplotlib.pyplot as plt
from PIL import Image
#二进制的形式读入
filename = '/Users/xuexin/Documents/GitHub/MNIST/MNIST_data/t10k-images-idx3-ubyte.gz'
binfile=open(filename,'rb')
buf=binfile.read()
binfile.close()

index=0
magic,images,rows,columns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')
print '开始循环rows=' + str(rows)
print '开始循环columns=' + str(columns)
for i in xrange(images):
    image = Image.new('L', (columns, rows))
    for x in xrange(rows):
        print '开始循环X'
        for y in xrange(columns):
            print '开始循环Y'
            image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
            index += struct.calcsize('>B')

    print 'save' + str(i) + 'image'
    image.save('test/' + str(i) + '.png')

#将每张图片按照格式存储到对应位置
# for image in range(0, images):
#     print 'start for'
#     im=struct.unpack_from('>784B',buf,index)
#     index+=struct.calcsize('>784B')
#     print '这里注意 Image对象的dtype是uint8，需要转换'
#     im=np.array(im,dtype='uint8')
#     im=im.reshape(28,28)
#     im=Image.fromarray(im)
#     print '保存图片index = %i'%index
#     im.save('/Users/xuexin/Documents/GitHub/MNIST/MNIST_data/train_%s.bmp'%image,'bmp')

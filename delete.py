import os

file = unicode("C:\Users\Administrator\Desktop\照片",'utf8')

files = os.listdir(file)

for i in files:       

    print i

    os.remove(os.path.join(file + "\\" + i,"1.jpg"))

可复制，其中 "\\"为路径间隔符，效果等同于os.path.join。

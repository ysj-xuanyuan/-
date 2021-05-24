import os

def ListFilesToTxt(dir,file,wildcard,recursion):
    for root, subdirs, files in os.walk(dir):
        subdirs.sort()
        print(subdirs)
        for i in range(len(subdirs)):
            #file.write(subdirs[i] + "\n")
            imgs= os.listdir(os.path.join(dir,subdirs[i]))
            imgs.sort()
            for j in range(len(imgs)):
                #print('1')
                file.write(subdirs[i] + '/'+imgs[j] + "\n")
        if(not recursion):
            break
def Test():
  dir="/home/test/Public/NTIRE2021/data/vsr/train/train_sharp"
  outfile="list.txt"
  wildcard = ".png"

  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)
  ListFilesToTxt(dir,file,wildcard,0)

  file.close()
Test()


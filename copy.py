import os
import shutil


def copy (path,num):
    for root ,folder_list ,file in os.walk(path):
        folder_list.sort()
        for i in range(0,30):
            print(folder_list[i])
            next_path=os.path.join(root,folder_list[i])
            print(next_path)
            file_list=os.listdir(next_path)
            file_list.sort()
            n = 0
            for file in file_list:
                
                if os.path.exists(next_path + '/'+"im_l.png"):
                    os.remove(os.path.join(next_path + '/'+"im_l.png"))
                if os.path.exists(next_path + '/'+"100.png"):
                    os.remove(os.path.join(next_path + '/'+"100.png"))
                oldname = next_path + os.sep + file 
                newname = next_path + os.sep + file.zfill(12) 
                os.rename(oldname, newname) 
                #print(oldname, '----->', newname)
                n=n+1
            file_list.sort()
            print(file_list[99])
            for j in range(1,91):
                
                if j%num==0:
                    j=j-1
                    file_path=os.path.join(next_path,file_list[j])
                    new_path=path+'/val_1/'+folder_list[i]+'_'+file_list[j]
                    print(file_path,new_path)
                    shutil.copyfile(file_path,new_path)
                
path='/home/test/ysj/STARnet-master/Results'     
copy(path,10)

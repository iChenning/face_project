# 读取
f_train = open(r'E:\list-zoo\train_widerface_list.txt')
origin = dict()
key = None
for line in f_train.readlines():
    line = line.replace('\n', '')
    if line.startswith('#'):
        if key is not None:
            origin[key] = info
        key = line
        info = []
    else:
        info.append(line)
f_train.close()

f_gt = open(r'E:\datasets\faces_widerface\wider_face_split\wider_face_train_bbx_gt.txt')
gt = dict()
key = None
for line in f_gt.readlines():
    line = line.replace('\n', '')
    if line.endswith('.jpg'):
        if key is not None:
            gt[key] = info
        key = line
        info = []
    else:
        if len(line) > 8:
            info.append(line)
f_gt.close()

# 将gt添加到原来的list中
new = dict()
for k_origin in origin.keys():
    for k_gt in gt.keys():
        if k_gt in k_origin:
            v_origin = origin[k_origin]
            v_gt = gt[k_gt]
            v_new = []
            for v_v_origin in v_origin:
                for v_v_gt in v_gt:
                    if v_v_gt[:-12] in v_v_origin:
                        v_v_new = v_v_origin + ' ' + v_v_gt[-12:]
                        v_new.append(v_v_new)
            new[k_origin] = v_new
            break

# save
f_save = open(r'E:\list-zoo\train-widerface_del-list.txt', 'w')
for key in new.keys():
    v_new = new[key]
    is_key = True
    for v in v_new:
        words = [float(x) for x in v.split()]
        if words[-2] == 2 or words[-6] == 2:
            continue
        else:
            if is_key:
                line = key + '\n'
                f_save.write(line)
                is_key = False
            line = v + '\n'
            f_save.write(line)
f_save.close()

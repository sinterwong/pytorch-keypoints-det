import json
import os
import glob
import shutil


def json2txt(data_path):
    json_files = glob.glob(data_path + '/*.json')
    if not os.path.exists('labels'):
        os.mkdir('labels')
    if not os.path.exists('images'):
        os.mkdir('images')
    for count, path in enumerate(json_files):
        file = os.path.basename(path)
        with open(os.path.join(data_path, file), 'r', encoding='gbk') as json_file:
            data = json.load(json_file)
            num_shapes = len(data['shapes'])
            str2write = ""
            str2write += (data['imagePath'])
            str2write += '\t' + str(num_shapes) + '\t'
            if num_shapes != 1:
                print(file)
                print(data['shapes'])
                continue
            for i, shape in enumerate(data['shapes']):
                if len(data['shapes'][i]['points']) == 4:
                    # 处理四边的情况，目前只需要两边
                    # x1 = data['shapes'][i]['points'][0][0]
                    # x2 = data['shapes'][i]['points'][1][0]
                    # x3 = data['shapes'][i]['points'][2][0]
                    # x4 = data['shapes'][i]['points'][3][0]
                    # y1 = data['shapes'][i]['points'][0][1]
                    # y2 = data['shapes'][i]['points'][1][1]
                    # y3 = data['shapes'][i]['points'][2][1]
                    # y4 = data['shapes'][i]['points'][3][1]
                    x1 = min(data['shapes'][i]['points'][0][0], data['shapes'][i]['points'][1][0], 
                            data['shapes'][i]['points'][2][0], data['shapes'][i]['points'][3][0])
                    x2 = max(data['shapes'][i]['points'][0][0], data['shapes'][i]['points'][1][0], 
                            data['shapes'][i]['points'][2][0], data['shapes'][i]['points'][3][0])
                    y1 = min(data['shapes'][i]['points'][0][1], data['shapes'][i]['points'][1][1], 
                            data['shapes'][i]['points'][2][1], data['shapes'][i]['points'][3][1])
                    y2 = max(data['shapes'][i]['points'][0][1], data['shapes'][i]['points'][1][1], 
                            data['shapes'][i]['points'][2][1], data['shapes'][i]['points'][3][1])
                elif len(data['shapes'][i]['points']) == 2:
                    # x1 = min(data['shapes'][i]['points'][0][0], data['shapes'][i]['points'][1][0])
                    # x3 = max(data['shapes'][i]['points'][0][0], data['shapes'][i]['points'][1][0])
                    # x2 = x3
                    # x4 = x1
                    # y1 = data['shapes'][i]['points'][0][1]
                    # y3 = data['shapes'][i]['points'][1][1]
                    # y2 = y3
                    # y4 = y1
                    x1 = min(data['shapes'][i]['points'][0][0], data['shapes'][i]['points'][1][0])
                    x2 = max(data['shapes'][i]['points'][0][0], data['shapes'][i]['points'][1][0])

                    y1 = min(data['shapes'][i]['points'][0][1], data['shapes'][i]['points'][1][1])
                    y2 = max(data['shapes'][i]['points'][0][1], data['shapes'][i]['points'][1][1])


                if int(x2 - x1) <= 5 or int(y2 - y1) <= 5:
                    flag = False
                    break
                if i != 0:
                    str2write += " " + str(x1)
                    str2write += " " + str(y1)
                    str2write += " " + str(x2)
                    str2write += " " + str(y2)
                else:
                    str2write += str(x1)
                    str2write += " " + str(y1)
                    str2write += " " + str(x2)
                    str2write += " " + str(y2)
            str2write += '\t'
            str2write += data['shapes'][i]['label']
        with open(os.path.join('labels', file.split('.')[0] + '_%s_%05d.txt' % (data['shapes'][i]['label'], count)), 'w', encoding='utf8') as txt_file:
                    txt_file.write(str2write)
        shutil.copy(os.path.join(
            data_path, file.replace('json', 'jpg')), os.path.join('images', file.split('.')[0] + '_%s_%05d.jpg' % (data['shapes'][i]['label'], count)))


if __name__ == "__main__":
    data_path = ""
    json2txt(data_path)

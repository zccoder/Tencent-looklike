from csv import DictWriter

with open('../../data_ori/userFeature.csv', 'w') as fo:
    headers = ['uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'interest1', 'interest2',
               'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',  'topic1', 'topic2', 'topic3', 'appIdInstall',
               'appIdAction', 'ct', 'os', 'carrier', 'house']
    writer = DictWriter(fo, fieldnames=headers, lineterminator='\n')
    writer.writeheader()

    fi = open('../../data_ori/userFeature.data', 'r')
    for t, line in enumerate(fi, start=1):
        line = line.replace('\n', '').split('|')
        userFeature_dict = {}
        for each in line:
            each_list = each.split(' ')
            userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
        writer.writerow(userFeature_dict)
        if t % 100000 == 0:
            print(t)
    fi.close()


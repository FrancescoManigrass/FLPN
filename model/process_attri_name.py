dir = './SUN_attri_name.txt'
with open(dir, 'r') as file:
    attri = file.readlines()
print(attri)
for item in attri:
    print(item)
with open(dir, 'w') as file:
    file.writelines(["%s\n" % item.replace('/', '_') for item in attri])
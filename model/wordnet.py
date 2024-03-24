import collections
import pickle

import nltk
import numpy as np
import scipy
from nltk.corpus import wordnet as wn
from tqdm import tqdm

dataset ="data/wgr/code/APN-ZSL-master/data/AWA2"

nltk.download('wordnet')
nltk.download('omw-1.4')

placental = wn.synset('placental.n.01')
seabird = wn.synset("bird.n.01")
hyper = lambda s: s.hypernyms()


def get_hyponyms(synset):
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))
    return hyponyms | set(synset.hyponyms())

prova = list(placental.closure(hyper, depth=20))
prova2 = list(seabird.closure(hyper, depth=50))
if "AWA2" in dataset:
    f = open(dataset+"/trainvalclasses.txt", "r")
    file = [x.rstrip().replace("+", "_") for x in f]
    Awa2_classes = [wn.synset(x.rstrip().replace("+", "_") + ".n.01") for x in file]

    f = open(dataset+"/testclasses.txt", "r")
    Awa2_classes_test = [wn.synset(x.rstrip().replace("+", "_") + ".n.01") for x in f]

    Awa2_classes.extend(Awa2_classes_test)
    Awa2_classes.remove(wn.synset("chihuahua.n.01"))
    Awa2_classes.append(wn.synset("chihuahua.n.03"))
    Awa2_classes.remove(wn.synset("dalmatian.n.01"))
    Awa2_classes.append(wn.synset("dalmatian.n.02"))
    Awa2_classes.remove(wn.synset("otter.n.01"))
    Awa2_classes.append(wn.synset("otter.n.02"))
    Awa2_classes.remove(wn.synset("gram_molecule.n.01"))
    Awa2_classes.append(wn.synset("mole.n.06"))
    Awa2_classes.remove(wn.synset("tiger.n.01"))
    Awa2_classes.append(wn.synset("tiger.n.02"))
    Awa2_classes.remove(wn.synset("leopard.n.01"))
    Awa2_classes.append(wn.synset("leopard.n.02"))
    Awa2_classes.remove(wn.synset("weasel.n.01"))
    Awa2_classes.append(wn.synset("weasel.n.02"))
    Awa2_classes.remove(wn.synset("raccoon.n.01"))
    Awa2_classes.append(wn.synset("raccoon.n.02"))

    Awa2_classes.remove(wn.synset("dolphinfish.n.02"))
    Awa2_classes.append(wn.synset("dolphin.n.02"))

    Awa2_classes.remove(wn.synset("sealing_wax.n.01"))
    Awa2_classes.append(wn.synset("seal.n.09"))

    Awa2_classes.remove(wn.synset("beaver.n.01"))
    Awa2_classes.append(wn.synset("beaver.n.07"))

    Awa2_classes.remove(wn.synset("rotter.n.01"))
    Awa2_classes.append(wn.synset("skunk.n.04"))


else:




    hypernums=[]
    hypernums.extend(get_hyponyms(wn.synset("seabird.n.01")))



    f = open(dataset + "/classes.txt", "r")
    file = [x.rstrip().split()[1] for x in f]
    file = [x.rstrip().split(".")[1]  for x in file]
    file = [x.lower().replace("_"," ") for x in file]
    file[0]=file[0].replace("black footed","black-footed")
    file[43]="frigate_bird"
    file[45] = "gad wall"
    #file[70] = "green_violetear"
    file = [x.lower() for x in file]
    Awa2_classes = []
    errors=[]
    for f in file:
        try:
            Awa2_classes.append(wn.synset(f.split(" ")[-1] + ".n.01"))
        except:
            errors.append(f)


    #Awa2_classes = [wn.synset(x.split(" ")[-1] + ".n.01") for x in file]

Awa2_gerarchia = {x: [] for x in Awa2_classes}



def iterate(x, initial):
    for x1 in x.hyponyms():
        if x1 in Awa2_classes:
            Awa2_gerarchia[x1].append(x)

            return
        else:
            iterate(x1, initial)


common = []
for f in Awa2_classes:
    for f1 in Awa2_classes:
        common.append(f1.lowest_common_hypernyms(f)[0])

most_common = collections.Counter(common).most_common()[0][0]
for i in tqdm(Awa2_classes):
    x = i

    while x != most_common:

        x = x.hypernyms()
        if isinstance(x, list):
            if x == []:
                break
        x = x[0]

        Awa2_gerarchia[i].append(x.name().split(".")[0])

att_splits = scipy.io.loadmat(dataset+"/binaryAtt_splits.mat")  # FOR AWA2
class_attributes_matrix = att_splits['att']
attributes_class_matrix = np.array(np.transpose(class_attributes_matrix))
classes_names = [f[0][0] for f in att_splits['allclasses_names']]

f = open(dataset+"/predicates.txt", "r")
Attributes_name = [x.split("\t")[1].rstrip() for x in f]
clean_gearchia = {f.name().split(".")[0].replace("_","+"): Awa2_gerarchia[f] for f in Awa2_gerarchia}

Categories = {f: clean_gearchia[f][-2] if len(clean_gearchia[f]) > 1 else clean_gearchia[f][-1] for f in clean_gearchia}


print("PRINTING KB")

KB={}
wordnet_dict = {}
for i in tqdm(range(len(attributes_class_matrix))):
    if classes_names[i]=="grizzly+bear":
        classes_names[i]="grizzly"
    if classes_names[i]=="polar+bear":
        classes_names[i]="ice+bear"
    if classes_names[i]=="moose":
        classes_names[i]="elk"
    if classes_names[i]=="buffalo":
        classes_names[i]="american+bison"
    if classes_names[i]=="humpback+whale":
        classes_names[i]="humpback"
    if classes_names[i]=="pig":
        classes_names[i]="hog"


    category_i = Categories[classes_names[i]]



    class_i = classes_names[i]
    attributes= [Attributes_name[f] for f in range(len(attributes_class_matrix[i])) if attributes_class_matrix[i][f]==1 ]
    attributes_index = [f for f in range(len(attributes_class_matrix[i])) if
                  attributes_class_matrix[i][f] == 1]
    if class_i in wordnet_dict:
        wordnet_dict[class_i]["macroclass"]=category_i
        wordnet_dict[class_i]["attributes"]=attributes
    else:
        wordnet_dict[class_i]={"macroclass":category_i,"attributes":attributes }

for i in tqdm(wordnet_dict.copy()):
    if i=="grizzly":
        tmp_dict= wordnet_dict[i]
        del wordnet_dict[i]
        wordnet_dict["grizzly+bear"]  = tmp_dict

    if i== "ice+bear":
        tmp_dict = wordnet_dict[i]
        del wordnet_dict[i]
        wordnet_dict["polar+bear"] = tmp_dict

    if i=="elk":

        tmp_dict = wordnet_dict[i]
        del wordnet_dict[i]
        wordnet_dict["moose"] = tmp_dict
    if i=="american+bison":

        tmp_dict = wordnet_dict[i]
        del wordnet_dict[i]
        wordnet_dict["buffalo"] = tmp_dict
    if i=="humpback" :
        tmp_dict = wordnet_dict[i]
        del wordnet_dict[i]
        wordnet_dict["humpback+whale"] = tmp_dict
    if i== "hog":
        tmp_dict = wordnet_dict[i]
        del wordnet_dict[i]
        wordnet_dict["pig"] = tmp_dict

print("save pickele")
with open(dataset+'/wordnet.pickle', 'wb') as handle:
    pickle.dump(wordnet_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)

with open(dataset+'/wordnet.pickle', 'rb') as f:
    x = pickle.load(f)
    print("dfdf")




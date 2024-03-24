import collections
import pickle

import nltk
import numpy as np
import scipy
from nltk.corpus import wordnet as wn
import scipy.io as sio

dataset ="data/wgr/code/APN-ZSL-master/data/CUB"

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
    file_new = []
    matcontent = sio.loadmat(dataset+"/" +"att_splits.mat")


    file = [x[0][0].split(".")[1] for x in matcontent['allclasses_names']]

    file = [x.rstrip().replace("_","-") for x in file]

    #file[file.index("Frigatebird")]= "Frigate_bird"
    #file[file.index("Northern_Waterthrush")] = "water_thrush"
    #file[file.index("Louisiana_Waterthrush")] = "water_thrush"
    Awa2_classes = {}
    errors=[]
    for f in file:
        res = [idx for idx in range(len(f)) if f[idx].isupper()]
        f=f.replace(f[res[-1]-1]+f[res[-1]],"_"+f[res[-1]])
        try:

            Awa2_classes[f]=wn.synset(f + ".n.01")
            file_new.append(f)
        except:


            try:
                if  "Waterthrush" in f:
                    Awa2_classes[f] = wn.synset("water_thrush" + ".n.01")
                elif  "Frigatebird" in f:
                    Awa2_classes[f] = wn.synset("Frigate_bird.n.01")
                elif  "Gadwall" in f:
                    Awa2_classes[f] = wn.synset("Anatidae.n.01")
                elif  'Green_Violetear' in f:
                    Awa2_classes[f] = wn.synset("Apodiformes.n.01")
                else:
                    Awa2_classes[f] = wn.synset(f.split("_")[-1] + ".n.01")
                    #print("dfdfd")
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
        common.append(Awa2_classes[f1].lowest_common_hypernyms(Awa2_classes[f])[0])

most_common = collections.Counter(common).most_common()[7][0]
macroclasses  = most_common.hyponyms()
#macroclasses[0] = macroclasses[0].

my_dict={}
my_dict2={}
my_dict3={}
my_dict4={}
for f in most_common.hyponyms():
    if f.hyponyms() != []:
        my_dict[f]=f.hyponyms()
    for j in my_dict[f]:
        if j.hyponyms() != []:
            my_dict2[j]=j.hyponyms()
            for j2 in my_dict2[j]:
                if j2.hyponyms() != []:
                    my_dict3[j2] = j2.hyponyms()
                    for j3 in my_dict3[j2]:
                        if j3.hyponyms() != []:
                            my_dict4[j3] = j3.hyponyms()

print("dfdf")
"""
for i in tqdm(Awa2_classes):
    x = Awa2_classes[i]
    print(i)
    while x.name().split(".")[0] != most_common.name().split(".")[0]:

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
"""
Awa2_classes= [f.replace("-","_") for f in Awa2_classes]
wordnet_dict={}
# seabird ->pleagic-bird ->albatross
for f in Awa2_classes:
    if "albatros" in f.lower():
        wordnet_dict[f.lower()]={"macroclass":"pelagic_bird"}
    if "tern" in f.lower():
        wordnet_dict[f.lower()]={"macroclass":"tern"}
    if "gull" in f.lower():
        wordnet_dict[f.lower()]={"macroclass":"gull"}
    if "grebe" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "grebe"}
    if "jaeger" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "jaeger"}
    if "auklet" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "auk"}
    if "guillemot" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "auk"}
    if "cormorant" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "pelecaniform_seabird"}
    if "kittiwake" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "gull"}
    if "frigate" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "pelecaniform_seabird"}
    if "puffin" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "puffin"}
    if "loon" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "loon"}


    if "pelican" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "pelecaniform_seabird"}

    if "penguin" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "sphenisciform_seabird"}

    if "blackbird" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "blackbird"} #passerine
    if "bobolink" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "bobolink"} #passerine
    if "bunting" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "bunting"}#passerine
    if "cardinal" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "cardinal"} #passerine
    if "cowbird" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "cowbird"}#passerine
    if "crow" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "passerine"}#passerine
    if "sparrow" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "sparrow"}
    if "thrasher" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "thrasher"}
    if "merganser" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "duck"}
    if "flycatcher" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "flycatcher"} #passerine
    if "tanager" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "tanager"} #passerine

    if "oriole" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "oriole"}#passerine
    if "raven" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "raven"}#passerine
    if "swallow" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "swallow"}#passerine
    if "warbler" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "warbler"}#passerine

    if "woodpecker" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "piciform_bird"}#passerine
    if "vireo" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "vireo"}#passerine

    if "kingfisher" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "kingfisher"}#passerine

    if "waxwing" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "waxwing"}#passerine
    if "wren" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "wren"}#passerine

    if "kingbird" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "tyrannid"}#passerine

    if "hummingbird" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "apodiform_bird"}#passerine
    if "catbird" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "catbird"}#passerine

    if "grosbeak" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "finch"}#passerine
    if "cuckoo" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "cuculiform_bird"}#passerine
    if "goldfinch" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "finch"}#passerine
    if "jay" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "corvine_bird"}#passerine

    if "nighthawk" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "caprimulgiform_bird"}#passerine
    if "yellow_breasted_chat" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "yellow_breasted_chat"}#passerine

    if "chuck_will_widow" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "chuck-will_widow"}#passerine
    if "brown_creeper" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "brown_creeper"}#passerine

    if "finch" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "finch"}#passerine

    if "flicker" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "piciform_bird"}#passerine

    if "fulmar" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "pelagic_bird"}  # passerine
    if "gadwall" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "gadwall"}  # passerine
    if "grackle" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "grackle"}  # passerine
    if "junco" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "finch"}  # passerine
    if "lark" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "lark"}  # passerine
    if "mallard" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "duck"}  # passerine
    if "nutcracker" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "nutcracker"}  # passerine
    if "nuthatch" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "nuthatch"}  # passerine
    if "ovenbird" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "warbler"}  # passerine
    if "sayornis" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "sayornis"}  # passerine
    if "pipit" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "pipit"}  # passerine
    if "junco" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "finch"}  # passerine
    if "lark" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "lark"}  # passerine
    if "redstart" in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "warbler"}  # passerine

    if 'geococcyx' in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "cuculiform_bird"}  # passerine
    if 'shrike' in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "shrike"}  # passerine

    if 'common_yellowthroat' in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "warbler"}  # passerine
    if 'waterthrush' in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "warbler"}  # passerine
    if 'green_tailed_towhee' in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "sparrow"}  # passerine

    if 'cape_Glossy_Starling'.lower() in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "starling"}
    if 'Whip_poor_Will'.lower() in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "starling"}
    if 'Green_Violetear'.lower() in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "Green_Violetear"}
    if 'Groove_billed_Ani'.lower() in f.lower():
        wordnet_dict[f.lower()] = {"macroclass": "Groove-billed_Ani"}









remained_class = [ f for f in Awa2_classes if f.lower() not in wordnet_dict]



print("dfdfd")
"""
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
"""
print("save pickele")
with open(dataset+'/wordnetCUB.pickle', 'wb') as handle:
    pickle.dump(wordnet_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)

with open(dataset+'/wordnetCUB.pickle', 'rb') as f:
    x = pickle.load(f)
    print("dfdf")




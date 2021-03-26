import os
from collections import defaultdict
from mmaction.core.evaluation.ava_utils import read_labelmap

cnt = defaultdict(int)
cats, _ = read_labelmap(open("data/ava/annotations/ava_action_list_v2.1.pbtxt"))
mAPs = """PerformanceByCategory/AP@0.5IOU/bend/bow (at the waist)=        0.3375637341679156
PerformanceByCategory/AP@0.5IOU/crawl=  0.012478954241880575
PerformanceByCategory/AP@0.5IOU/crouch/kneel=   0.16025440783784484
PerformanceByCategory/AP@0.5IOU/dance=  0.4612299621746703
PerformanceByCategory/AP@0.5IOU/fall down=      0.07525531834079713
PerformanceByCategory/AP@0.5IOU/get up= 0.11285021874922746
PerformanceByCategory/AP@0.5IOU/jump/leap=      0.01887828542783989
PerformanceByCategory/AP@0.5IOU/lie/sleep=      0.4779084757504234
PerformanceByCategory/AP@0.5IOU/martial art=    0.37881131883605457
PerformanceByCategory/AP@0.5IOU/run/jog=        0.24354705149701125
PerformanceByCategory/AP@0.5IOU/sit=    0.7708792875236453
PerformanceByCategory/AP@0.5IOU/stand=  0.7832942789090698
PerformanceByCategory/AP@0.5IOU/swim=   0.3687255293750998
PerformanceByCategory/AP@0.5IOU/walk=   0.6433761010283656
PerformanceByCategory/AP@0.5IOU/answer phone=   0.708238415223992
PerformanceByCategory/AP@0.5IOU/brush teeth=    0.00020992912086879338
PerformanceByCategory/AP@0.5IOU/carry/hold (an object)= 0.5127380826977359
PerformanceByCategory/AP@0.5IOU/catch (an object)=      0.0002463393092275414
PerformanceByCategory/AP@0.5IOU/chop=   0.0003051665635858549
PerformanceByCategory/AP@0.5IOU/climb (e.g., a mountain)=       0.013023644900349095
PerformanceByCategory/AP@0.5IOU/clink glass=    0.0003696811427095587
PerformanceByCategory/AP@0.5IOU/close (e.g., a door, a box)=    0.09945402454261718
PerformanceByCategory/AP@0.5IOU/cook=   0.0014739411811829587
PerformanceByCategory/AP@0.5IOU/cut=    0.039373057893932244
PerformanceByCategory/AP@0.5IOU/dig=    0.03275001993051673
PerformanceByCategory/AP@0.5IOU/dress/put on clothing=  0.01739219692708466
PerformanceByCategory/AP@0.5IOU/drink=  0.19370596924404151
PerformanceByCategory/AP@0.5IOU/drive (e.g., a car, a truck)=   0.5320579668277279
PerformanceByCategory/AP@0.5IOU/eat=    0.22757444087929363
PerformanceByCategory/AP@0.5IOU/enter=  0.03136868102076874
PerformanceByCategory/AP@0.5IOU/exit=   0.0008599062818156537
PerformanceByCategory/AP@0.5IOU/extract=        0.00022446847192553526
PerformanceByCategory/AP@0.5IOU/fishing=        0.057228251708354404
PerformanceByCategory/AP@0.5IOU/hit (an object)=        0.001471196280949075
PerformanceByCategory/AP@0.5IOU/kick (an object)=       0.0001923047337733266
PerformanceByCategory/AP@0.5IOU/lift/pick up=   0.016821042445646736
PerformanceByCategory/AP@0.5IOU/listen (e.g., to music)=        0.008185812849057526
PerformanceByCategory/AP@0.5IOU/open (e.g., a window, a car door)=      0.14137897563492519
PerformanceByCategory/AP@0.5IOU/paint=  0.00010490449494779953
PerformanceByCategory/AP@0.5IOU/play board game=        0.0006410273681142254
PerformanceByCategory/AP@0.5IOU/play musical instrument=        0.17764566004504603
PerformanceByCategory/AP@0.5IOU/play with pets= 0.0015387226193109417
PerformanceByCategory/AP@0.5IOU/point to (an object)=   0.0005577375470210308
PerformanceByCategory/AP@0.5IOU/press=  0.001736250534542612
PerformanceByCategory/AP@0.5IOU/pull (an object)=       0.007793085972804026
PerformanceByCategory/AP@0.5IOU/push (an object)=       0.013875685550341878
PerformanceByCategory/AP@0.5IOU/put down=       0.027836413045180773
PerformanceByCategory/AP@0.5IOU/read=   0.24070285443058248
PerformanceByCategory/AP@0.5IOU/ride (e.g., a bike, a car, a horse)=    0.3255960156859441
PerformanceByCategory/AP@0.5IOU/row boat=       0.023871068991638964
PerformanceByCategory/AP@0.5IOU/sail boat=      0.10048398690255074
PerformanceByCategory/AP@0.5IOU/shoot=  0.03643798005503207
PerformanceByCategory/AP@0.5IOU/shovel= 0.18201475646205978
PerformanceByCategory/AP@0.5IOU/smoke=  0.11373454709873781
PerformanceByCategory/AP@0.5IOU/stir=   0.0007442537790334074
PerformanceByCategory/AP@0.5IOU/take a photo=   0.002269972377905052
PerformanceByCategory/AP@0.5IOU/text on/look at a cellphone=    0.03493897423477036
PerformanceByCategory/AP@0.5IOU/throw=  0.007721224421836978
PerformanceByCategory/AP@0.5IOU/touch (an object)=      0.2639093835786261
PerformanceByCategory/AP@0.5IOU/turn (e.g., a screwdriver)=     0.002342118884304465
PerformanceByCategory/AP@0.5IOU/watch (e.g., TV)=       0.2615145135266727
PerformanceByCategory/AP@0.5IOU/work on a computer=     0.05294945291797928
PerformanceByCategory/AP@0.5IOU/write=  0.052582466378676494
PerformanceByCategory/AP@0.5IOU/fight/hit (a person)=   0.2950941950980064
PerformanceByCategory/AP@0.5IOU/give/serve (an object) to (a person)=   0.042314752563796104
PerformanceByCategory/AP@0.5IOU/grab (a person)=        0.0536426919579943
PerformanceByCategory/AP@0.5IOU/hand clap=      0.09892780215091958
PerformanceByCategory/AP@0.5IOU/hand shake=     0.04420566796916442
PerformanceByCategory/AP@0.5IOU/hand wave=      0.006838782301720842
PerformanceByCategory/AP@0.5IOU/hug (a person)= 0.15657911611695075
PerformanceByCategory/AP@0.5IOU/kick (a person)=        0.00043486501796337535
PerformanceByCategory/AP@0.5IOU/kiss (a person)=        0.24560608842988557
PerformanceByCategory/AP@0.5IOU/lift (a person)=        0.022272692098608604
PerformanceByCategory/AP@0.5IOU/listen to (a person)=   0.5107434744480743
PerformanceByCategory/AP@0.5IOU/play with kids= 0.0009366900429949059
PerformanceByCategory/AP@0.5IOU/push (another person)=  0.015555311707819812
PerformanceByCategory/AP@0.5IOU/sing to (e.g., self, a person, a group)=        0.08015347379406504
PerformanceByCategory/AP@0.5IOU/take (an object) from (a person)=       0.02433625754376946
PerformanceByCategory/AP@0.5IOU/talk to (e.g., self, a person, a group)=        0.6986897304633258
PerformanceByCategory/AP@0.5IOU/watch (a person)=       0.6354140496544081"""

with open("/ssd01/zhangyiyang/mmaction2_github/data/ava/annotations/ava_train_v2.1.csv", "r") as f:
    for line in f:
        cnt[int(line.split(",")[-2])] += 1
mAP = [float(line.split(" ")[-1]) for line in mAPs.split("\n")]

samples = []
res = .0
num = 0
ids = []
for i in range(1, 81):
    # print(cats[i-1]['name'], cnt[i], round(mAP[i-1], 4))
    # print(f"|{cats[i-1]['name']}|{cnt[i]}|{round(mAP[i-1], 4)}|")
    # samples.append([cats[i-1]['name'], cnt[i], round(mAP[i-1], 4)])
    # if .6 > mAP[i-1] >= .3:
    if 100 > cnt[i] > 0:
        res += mAP[i-1]
        num += 1
        ids.append(str(i))

print(res / num, num)
print(",".join(ids))

# samples = sorted(samples, key=lambda k: -k[2])

# print("|class|cnt of samples|AP|")
# print("|:-:|:-:|:-:|")
# for sample in samples:
#     print(f"|{sample[0]}|{sample[1]}|{sample[2]}|")

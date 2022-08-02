# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import numpy as np
import json
from pathlib import Path

class Result(object):
    def __init__(self):
        self.res_dict = {
            'det': dict(),
            'mot': dict(),
            'attr': dict(),
            'kpt': dict(),
            'action': dict(),
            'reid': dict()
        }

    def update(self, res, name):
        self.res_dict[name].update(res)

    def get(self, name):
        if name in self.res_dict and len(self.res_dict[name]) > 0:
            return self.res_dict[name]
        return None

    def clear(self, name):
        self.res_dict[name].clear()
    def export(self):
        #we have to deepcopy since many values are shared between frames
        return copy.deepcopy( self.res_dict)
    def save(self, fn):
        with open(fn, 'wb') as f:
            np.save( f, self.res_dict)
    def load( self, fn):
        with open(fn, 'rb') as f:
            self.res_dict = np.load( f, allow_pickle=True)

def npdefault(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            if obj.dtype in [ np.dtype('float64'), np.dtype('float32')]:
                pril = np.array( obj,int).tolist()
            else:
                pril = obj.tolist()
            return pril
        else:
            return obj.item()
        raise TypeError('Unknown type:', type(obj))

def fixkpt( kpt):
    mpkps = kpt['keypoint']
    #its dimention is 3.  frame ->  person -> point -> coordinate
    mpkps = [[ np.array( pkp,int) for pkp in pkps] for pkps in mpkps]
    kpt['keypoint'] = mpkps

def fixmot( mot):
    if type(mot['boxes'])==list:
        return
    #overwrite original 'boxes'
    [ids, scores, boxes] = [[], [], []]
    for obox in mot['boxes']:
        bx = [int(b) for b in obox]
        ids.append( bx[0])
        scores.append(  bx[1])
        boxes.append( bx[3:])
    mot['boxes'] = boxes
    return ids
    #mot['scores'] = scores

def pattr2dic( atrs):
    """
    [
        "Male",
        "Age18-60",
        "Side",
        "Glasses: False",
        "Hat: False",
        "HoldObjectsInFront: False",
        "No bag",
        "Upper: LongSleeve",
        "Lower:  Trousers",
        "No boots"
    ],
    """
    return {
        'gender': atrs[0],
        'age': atrs[1][3:],
        'side': atrs[2]=='Side',
        'glasses': atrs[3].split(': ')[1]=='True',
        'hat': atrs[4].split(': ')[1]=='True',
        'holdobjectsinfront': atrs[5].split(': ')[1]=='True',
        'bag': atrs[6].split(' ')[0]!='No',
        'upper': atrs[7].split(': ')[1],
        'lower': atrs[8].split(': ')[1],
        'boots': atrs[9].split(' ')[0]!="No"
    }

def fixattr( rawatr):
    pattrs = rawatr['output']
    pdics = [ pattr2dic(pattr) for pattr in pattrs ]
    rawatr['people'] = pdics
    del rawatr['output']

class DataCollector(object):
    """
  DataCollector of pphuman Pipeline, collect results in every frames and assign it to each track ids.
  mainly used in mtmct.
  
  data struct:
  collector:
    - [id1]: (all results of N frames)
      - frames(list of int): Nx[int]
      - rects(list of rect): Nx[rect(conf, xmin, ymin, xmax, ymax)]
      - features(list of array(256,)): Nx[array(256,)]
      - qualities(list of float): Nx[float]
      - attrs(list of attr): refer to attrs for details
      - kpts(list of kpts): refer to kpts for details
      - actions(list of actions): refer to actions for details
    ...
    - [idN]
  """

    def __init__(self):
        #id, frame, rect, score, label, attrs, kpts, actions
        self.mots = {
            "frames": [],
            "rects": [],
            "attrs": [],
            "kpts": [],
            "features": [],
            "qualities": [],
            "actions": []
        }
        self.collector = {}
        self.frame_results = []

    def append(self, frameid, Result):
        self.frame_results.append(Result.export())
        mot_res = Result.get('mot')
        attr_res = Result.get('attr')
        kpt_res = Result.get('kpt')
        action_res = Result.get('action')
        reid_res = Result.get('reid')
        rects = reid_res['rects'] if reid_res is not None else mot_res['boxes']
        for idx, mot_item in enumerate(rects):
            ids = int(mot_item[0])
            if ids not in self.collector:
                self.collector[ids] = copy.deepcopy(self.mots)
            self.collector[ids]["frames"].append(frameid)
            self.collector[ids]["rects"].append([mot_item[2:]])
            if attr_res:
                self.collector[ids]["attrs"].append(attr_res['output'][idx])
            if kpt_res:
                self.collector[ids]["kpts"].append(
                    [kpt_res['keypoint'][0][idx], kpt_res['keypoint'][1][idx]])
            if action_res and (idx + 1) in action_res:
                self.collector[ids]["actions"].append(action_res[idx + 1])
            else:
                # action model generate result per X frames, Not available every frames
                self.collector[ids]["actions"].append(None)
            if reid_res:
                self.collector[ids]["features"].append(reid_res['features'][
                    idx])
                self.collector[ids]["qualities"].append(reid_res['qualities'][
                    idx])
    #ID based result
    def get_res(self):
        return self.collector

    def _merge_extra( self, infrfrms):
        keys = [ 'inftype', 'width', 'height', 'fps', 'frame_count', 'entrance_line']
        vidinf = { k:self.exinf[k] for k in self.exinf.keys() if k in keys }
        for ( exfrm, infrfrm) in zip( self.exinf['frames'], infrfrms):
            if 'entrance' in exfrm:
                infrfrm['entrance'] = exfrm['entrance']
        vidinf['frames'] = infrfrms
        return vidinf
    def save_frame_res(self, dstfn):
        frmres = self.frame_results
        frminfs=[]
        for frminf in frmres:
            fixkpt( frminf['kpt'])
            ids = fixmot( frminf['mot'])
            fixattr( frminf['attr'])
            frminf['ids'] = ids
            frminfs.append(frminf)
        vidinf = self._merge_extra( frminfs)
        jobj = json.dumps( vidinf, indent = 4, default=npdefault)
        with open( dstfn, 'w') as outfile:
            outfile.write(jobj)
    def set_extra_info(self, exinf):
        self.exinf = exinf

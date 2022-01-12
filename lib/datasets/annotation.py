import xml.etree.ElementTree as ET
import copy
import numpy as np

class Annotation:
    def __init__(self, path=None):
        self.path = path
        self.anno = {}
        if self.path is not None:
            self.get_anno()
        

    def get_anno(self):
        def dfs(d, root):
            for child in root:
                if len(child) == 0:
                    d[child.tag] = child.text if not child.text.isnumeric() else np.float32(child.text)
                else:
                    d[child.tag] = {}
                    dfs(d[child.tag], child)
        self.anno={}
        tree = ET.parse(self.path)
        root = tree.getroot()
        dfs(self.anno, root)

    def lookup(self, key):
        # return a copy of first instance of key in the annotation by performing a dfs search.
        all = []
        def dfs(d, key, lst):
            for k, v in d.items():
                if k == key:
                    lst.append(copy.deepcopy(v))
                if isinstance(v, dict):
                    dfs(v, key, lst)
            
        dfs(self.anno, key, all)
        return all[0] if len(all) > 0 else None

    def get_bbox(self, tag='headbndbox'):
        four_corners = ['xmin', 'ymin', 'xmax', 'ymax']
        return [self.lookup(tag)[corner] for corner in four_corners]

    def update(self, key, value):
        all = []
        def dfs(d, key, lst):
            for k, v in d.items():
                if k == key:
                    lst.append(d)
                if isinstance(v, dict):
                    dfs(v, key, lst)
            
        dfs(self.anno, key, all)
        node = all[0]
        assert node is not None, '{} does not exists'.format(key)
        node[key] = value

# -*- coding: utf-8 -*-
import numpy as np
import ctypes
from OpenGL.GL import *

from .ebo import EBO


class VAO(object):
    def __init__(self, vbo_attrib, ebo=None):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateVertexArrays(len(self.__id), self.__id)
        i = 0
        for vbo_offset_stride, attribs in vbo_attrib.items():
            vbo = vbo_offset_stride[0]
            offset = vbo_offset_stride[1]
            stride = vbo_offset_stride[2]
            for attrib in attribs:
                attribindex = attrib[0]
                size = attrib[1]
                attribtype = attrib[2]
                normalized = attrib[3]
                relativeoffset = attrib[4]
                glVertexArrayAttribFormat(self.__id[0], attribindex, size, attribtype, normalized, relativeoffset)
                glVertexArrayAttribBinding(self.__id[0], attribindex, i)
                glEnableVertexArrayAttrib(self.__id[0], attribindex)
            glVertexArrayVertexBuffer(self.__id[0], i, vbo.id[0], offset, stride)
            i += 1
        if ebo != None:
            if isinstance(ebo, EBO):
                glVertexArrayElementBuffer(self.__id[0], ebo.id[0])
            else:
                ValueError("Invalid EBO type.")

    def bind(self):
        glBindVertexArray(self.__id[0])

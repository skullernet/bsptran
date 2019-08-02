#!/usr/bin/env python3

import struct
import sys
import re
import numpy as np
import math
import io

matrix = np.eye(3, 3)
matrix_inv = np.eye(3, 3)
matrix_det = 1

def cross_matrix(x, y, z):
    return np.array([
        [ 0, -z,  y ],
        [ z,  0, -x ],
        [-y,  x,  0 ]
    ])

def rot_matrix(axis, theta):
    s = math.sin(theta * math.pi / 180)
    c = math.cos(theta * math.pi / 180)
    return c * np.eye(3, 3) + s * cross_matrix(*axis) + (1 - c) * np.outer(axis, axis)

def scale_matrix(axis, scale):
    m = np.eye(3, 3)
    m[axis][axis] = scale
    return m

def transform_plane(norm, dist):
    point = matrix @ (norm * dist)
    norm = norm @ matrix_inv
    norm /= np.linalg.norm(norm)
    dist = point @ norm
    return norm, dist

snap_axis = np.array([np.eye(3, 3), -np.eye(3, 3)])
def snap_normal(norm):
    for i in 0, 1, 2:
        for j in 0, 1:
            v = snap_axis[j][i]
            if np.allclose(norm, v, rtol=0, atol=1e-05):
                v = np.array(v, dtype=np.dtype(int))
                return v, (i, 6)[j]
    return norm, 6

def snap_dist(dist):
    d = round(dist)
    if abs(dist - d) < 0.01:
        return d
    return dist

def do_plane(data):
    for i, j in enumerate(struct.iter_unpack('<ffffi', data)):
        norm, dist = transform_plane(np.array(j[:3]), j[3])
        norm, typ = snap_normal(norm)
        struct.pack_into('<ffffi', data, i * 20, *norm, snap_dist(dist), typ)

def do_vertex(data):
    for i, j in enumerate(struct.iter_unpack('<fff', data)):
        pos = matrix @ np.array(j)
        struct.pack_into('<fff', data, i * 12, *pos)

surfedges = []
def read_surfedges(f, hdr):
    ofs, cnt = hdr
    data = bytearray(cnt)
    f.seek(ofs)
    f.readinto(data)

    for i in struct.iter_unpack('<i', data):
        surfedges.append(i[0])

def write_surfedges(o, hdr):
    ofs, cnt = hdr
    data = bytearray(cnt)

    for i, j in enumerate(surfedges):
        struct.pack_into('<i', data, i * 4, j)

    o.seek(ofs)
    o.write(data)

def do_faces(data):
    if matrix_det > 0:
        return
    for i in struct.iter_unpack('<HHIHHBBBBI', data):
        start = i[2]
        end = start + i[3]
        surfedges[start:end] = reversed(surfedges[start:end])

def do_edges(data):
    if matrix_det > 0:
        return
    for i, j in enumerate(struct.iter_unpack('<HH', data)):
        struct.pack_into('<HH', data, i * 4, j[1], j[0])

def do_texinfo(data):
    fmt = '<8fIi32sI'
    size = struct.calcsize(fmt)
    for i, j in enumerate(struct.iter_unpack(fmt, data)):
        st = np.array([j[0:3], j[4:7]]) @ matrix_inv
        struct.pack_into(fmt, data, i * size, *st[0], j[3], *st[1], *j[7:])

def expand_bounds(bounds, dtype=None):
    points = np.empty((8, 3))
    for i in range(0, 8):
        p = np.array([
            bounds[(i >> 0) & 1][0],
            bounds[(i >> 1) & 1][1],
            bounds[(i >> 2) & 1][2]
        ])
        points[i] = matrix @ p
    return np.array([
        np.amin(points, axis=0),
        np.amax(points, axis=0)], dtype=dtype)

def do_nodes(data):
    fmt = '<3I6hHH'
    size = struct.calcsize(fmt)
    for i, j in enumerate(struct.iter_unpack(fmt, data)):
        b = expand_bounds(np.array([j[3:6], j[6:9]]), np.dtype(int))
        struct.pack_into(fmt, data, i * size, *j[:3], *b[0], *b[1], *j[9:])

def do_leafs(data):
    fmt = '<IHH6h4H'
    size = struct.calcsize(fmt)
    for i, j in enumerate(struct.iter_unpack(fmt, data)):
        b = expand_bounds(np.array([j[3:6], j[6:9]]), np.dtype(int))
        struct.pack_into(fmt, data, i * size, *j[:3], *b[0], *b[1], *j[9:])

def do_models(data):
    fmt = '<9f3I'
    size = struct.calcsize(fmt)
    for i, j in enumerate(struct.iter_unpack(fmt, data)):
        bounds = expand_bounds(np.array([j[0:3], j[3:6]]))
        origin = matrix @ np.array(j[6:9])
        struct.pack_into(fmt, data, i * size, *bounds[0], *bounds[1], *origin, *j[9:])

def tokenizer(s):
    strpos = 0
    whitespace = re.compile(r'\s+')
    comment = re.compile(r'//.*?\n')
    quoted = re.compile(r'".*?"')
    word = re.compile(r'\S+')
    while strpos < len(s):
        m = whitespace.match(s, strpos)
        if m:
            strpos = m.end() + 1
            continue
        m = comment.match(s, strpos)
        if m:
            strpos = m.end() + 1
            continue
        m = quoted.match(s, strpos)
        if m:
            yield m.group()[1:-1]
            strpos = m.end() + 1
            continue
        m = word.match(s, strpos)
        if m:
            yield m.group()
            strpos = m.end() + 1
            continue
        raise ValueError()

def anglestovec(val):
    p = val[0] * math.pi / 180
    y = val[1] * math.pi / 180
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    return np.array([cp * cy, cp * sy, -sp])

def vectoangles(val):
    if abs(val[1]) < 1e-05 and abs(val[0]) < 1e-05:
        yaw = 0
        pitch = 90 if val[2] > 0 else 270
    else:
        if abs(val[0]) > 1e-05:
            yaw = math.atan2(val[1], val[0]) / math.pi * 180
        elif val[1] > 0:
            yaw = 90
        else:
            yaw = -90
        if yaw < 0:
            yaw += 360

        fwd = math.sqrt(val[0] * val[0] + val[1] * val[1])
        pitch = -math.atan2(val[2], fwd) / math.pi * 180
        if pitch < 0:
            pitch += 360

    return np.array([pitch, yaw, 0])

def transform_angles(val):
    return vectoangles(matrix @ anglestovec(val))

def transform_angle(val):
    return vectoangles(matrix @ anglestovec([0, val, 0]))[1]

def do_entities(f, o, hdr):
    ofs, cnt = hdr
    f.seek(ofs)
    tok = tokenizer(f.read(cnt).rstrip(b'\x00').decode('ascii', errors='surrogateescape'))

    r = ''
    for s in tok:
        if s != '{':
            raise ValueError()
        r += '{\n'
        for s in tok:
            if s == '}':
                break
            key = s
            val = next(tok)
            if key == 'origin':
                val = matrix @ np.fromstring(val, sep=' ')
                val = '%.f %.f %.f' % tuple(val)
            elif key == 'angles':
                val = transform_angles(np.fromstring(val, sep=' '))
                val = '%.f %.f %.f' % tuple(val)
            elif key == 'angle':
                val = transform_angle(float(val))
                val = '%.f' % val
            r += '"%s" "%s"\n' % (key, val)
        r += '}\n'

    o.seek(0, io.SEEK_END)
    pos = o.tell()
    r = r.encode('ascii', errors='surrogateescape')
    o.write(r)
    o.seek(8)
    o.write(struct.pack('<II', pos, len(r)))

def process(f, o, hdr, func=None):
    ofs, cnt = hdr
    data = bytearray(cnt)
    f.seek(ofs)
    f.readinto(data)
    if func:
        func(data)
    o.seek(ofs)
    o.write(data)

def main():
    global matrix
    global matrix_inv
    global matrix_det

    def usage():
        sys.exit(f'Usage: {sys.argv[0]} [options] <input.bsp> <output.bsp>')

    opts = {
        '-s': (lambda: np.eye(3, 3) * a),
        '-sx': (lambda: scale_matrix(0, a)),
        '-sy': (lambda: scale_matrix(1, a)),
        '-sz': (lambda: scale_matrix(2, a)),
        '-rx': (lambda: rot_matrix([1, 0, 0], a)),
        '-ry': (lambda: rot_matrix([0, 1, 0], a)),
        '-rz': (lambda: rot_matrix([0, 0, 1], a)),
    }

    i = 1
    while i < len(sys.argv):
        c = sys.argv[i]
        if c == '-h':
            usage()
        if c[0] != '-':
            break
        if not c in opts:
            sys.exit(f'Unknown option: {c}')
        a = float(sys.argv[i+1])
        matrix = opts[c]() @ matrix
        i += 2

    if i+2 > len(sys.argv):
        usage()

    matrix_inv = np.linalg.inv(matrix)
    matrix_det = np.linalg.det(matrix)

    with open(sys.argv[i  ], mode='rb') as f, \
         open(sys.argv[i+1], mode='wb') as o:

        hdr = f.read(20*8)
        if struct.unpack('<4si', hdr[:8]) != (b'IBSP', 38):
            sys.exit('Not an IBSP version 38')

        o.write(hdr)

        hdr = list(struct.iter_unpack('<II', hdr[8:]))

        read_surfedges(f, hdr[12])

        do = {
            1: do_plane, 2: do_vertex, 4: do_nodes,  5: do_texinfo,
            6: do_faces, 8: do_leafs, 11: do_edges, 13: do_models
        }

        for k, v in do.items():
            process(f, o, hdr[k], v)

        for i in 3, 7, 9, 10, 14, 15, 17, 18:
            process(f, o, hdr[i])

        write_surfedges(o, hdr[12])

        do_entities(f, o, hdr[0])

if __name__ == '__main__':
    main()

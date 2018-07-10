from enum import Enum

class Point2D(object):
    def __init__(self, x: float = 0.0, y: float = 0.0) -> object:
        self.x = x  # type: float
        self.y = y  # type: float

    def __add__(self, p):  # p is Point2D
        return Point2D(self.x + p.x, self.y + p.y)

    def __sub__(self, p):  # p is Point2D
        return Point2D(self.x - p.x, self.y - p.y)

    def __truediv__(self, op: float):
        return Point2D(self.x / op, self.y / op)

    def __mul__(self, op: float):
        return Point2D(self.x * op, self.y * op)


class LineSegment(object):
    def __init__(self, startPos=Point2D(), endPos=Point2D()):
        self.startPos = startPos
        self.endPos = endPos
        self.type = ""
        self.portalToRoom = ""


class Space(object):
    def __init__(self):
        self.category = ""
        self.vertex_id = ""
        self.roomLayout = []  # type: [LineSegment]
        self.maxx = 0.0
        self.maxy = 0.0
        self.minx = 0.0
        self.miny = 0.0
        self.centroid = Point2D()

    def updateExtent(self):
        self.maxx = float("-inf")
        self.maxy = float("-inf")
        self.minx = float("inf")
        self.miny = float("inf")

        for line_seg in self.roomLayout:
            cur_minx = min(line_seg.startPos.x, line_seg.endPos.x)
            cur_miny = min(line_seg.startPos.y, line_seg.endPos.y)
            cur_maxx = max(line_seg.startPos.x, line_seg.endPos.x)
            cur_maxy = max(line_seg.startPos.y, line_seg.endPos.y)

            if cur_minx < self.minx:
                self.minx = cur_minx

            if cur_miny < self.miny:
                self.miny = cur_miny

            if cur_maxx > self.maxx:
                self.maxx = cur_maxx

            if cur_maxy > self.maxy:
                self.maxy = cur_maxy


class EdgeClass(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


class EdgeType(Enum):
    EXPLICIT_EDGE = 1
    IMPLICIT_EDGE = 2


class spaceEdge(object):
    def __init__(self):
        self.edge_id = ""
        self.edge_class = EdgeClass.HORIZONTAL
        self.edge_type = EdgeType.EXPLICIT_EDGE


class GraphProperties(object):
    def __init__(self):
        self.floorname = ""
        self.filepath = ""
        self.maxx = 0.0
        self.maxy = 0.0
        self.minx = 0.0
        self.miny = 0.0
        self.centroid = Point2D()
        self.pixel_distance = 0.0
        self.real_distance = 0.0
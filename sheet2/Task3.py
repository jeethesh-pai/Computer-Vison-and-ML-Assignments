import math


class Vec2:
    global_id = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.variable_id = Vec2.global_id + 1
        Vec2.global_id += 1

    def length(self):
        """calculates the length of a vector given origin
        :returns
        distance:- euclidean distance from the origin"""
        distance = math.sqrt(self.x**2 + self.y**2)
        return distance

    def add(self, rhs):
        """calculates sum of two vectors given another vector
        :param
        rhs :- vector of type Vec2
        :returns
        new vector of type Vec2"""
        return Vec2((rhs.x + self.x), (rhs.y + self.y))


if __name__ == '__main__':
    vec1 = Vec2(5, 6)
    print('length of the vector: ', vec1.length())
    vec2 = Vec2(7, 8)
    vec3 = vec1.add(vec2)
    print('sum of vec1 and vec2 is: ', vec3.x, ' ', vec3.y)

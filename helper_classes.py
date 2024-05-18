import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    # TODO:
    v = np.array([0, 0, 0])
    return v

# Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(direction)

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection_point):
        return Ray(intersection_point, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = position
        self.direction = direction
        self.kc = kc
        self.kl = kl
        self.kq = kq


    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        ray = normalize(intersection - self.position)
        d = self.get_distance_from_light(intersection)
        fatt = (self.kq * d**2) + (self.kl * d) + self.kc
        return self.intensity * (np.dot(self.direction,normalize(ray)) / fatt)


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.shininess = shininess
        self.reflection = reflection


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects) -> tuple[int, Object3D]:
        intersections = []
        for object in objects:
            intersection = object.intersect(self)
            if intersection and intersection[0] > 0:
                intersections.append(intersection)
        if not intersections:
            return None
        else:
            min_intersect = min(intersections, key=lambda x: x[0])
            return min_intersect

    def find_point_with_given_t(self, t: int):
        return self.origin + t * self.direction


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / \
            (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None

    def get_normal(self, intersection_point):
        return self.normal


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.

    """

    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.get_normal(0)

    # computes normal to the trainagle surface. Pay attention to its direction!
    def get_normal(self,intersection_point):
        AB = self.b - self.a
        AC = self.c - self.a
        return normalize(np.cross(AB, AC))

    # This function returns the intersection point of the ray with the triangle
    def intersect(self, ray):
        edge_ab = self.b - self.a
        edge_ac = self.c - self.a

        direction_cross_ac = np.cross(ray.direction, edge_ac)
        denominator = np.dot(edge_ab, direction_cross_ac)

        if -1e-3 < denominator < 1e-3:
            return None

        inverse_denominator = 1.0 / denominator
        origin_to_a = ray.origin - self.a

        alpha = inverse_denominator * np.dot(origin_to_a, direction_cross_ac)
        if alpha < 0.0 or alpha > 1.0:
            return None

        origin_cross_ab = np.cross(origin_to_a, edge_ab)
        beta = inverse_denominator * np.dot(ray.direction, origin_cross_ab)
        if beta < 0.0 or alpha + beta > 1.0:
            return None

        t = inverse_denominator * np.dot(edge_ac, origin_cross_ab)
        if t < 1e-3:
            return None

        return t, self
            



class Pyramid(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 

    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """

    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self) :
        l = []
        t_idx = [
                [0, 1, 3],
                [1, 2, 3],
                [0, 3, 2],
                [4, 1, 0],
                [4, 2, 1],
                [2, 4, 0]]
        for idx in t_idx:
            l.append(Triangle(self.v_list[idx[0]],self.v_list[idx[1]],self.v_list[idx[2]]))
        return l

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)

    def intersect(self, ray: Ray):
        return ray.nearest_intersected_object(self.triangle_list)

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        # Step 1: Define variables
        L = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, L)
        c = np.dot(L, L) - self.radius ** 2

        # Step 2: Calculate the discriminant
        discriminant = b ** 2 - 4 * a * c

        # Step 3: Determine if there is an intersection based on the discriminant
        if discriminant < 0:
            return None  # No real roots; no intersection
        elif discriminant == 0:
            # One solution (tangent to the sphere)
            t = -b / (2 * a)
            if t < 0:
                return None  # Intersection is behind the ray's origin
            intersection_point = ray.origin + t * ray.direction
            return intersection_point
        else:
            # Two solutions (the ray passes through the sphere)
            sqrt_discriminant = np.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)

            # Only consider intersections in the direction the ray is pointing
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
            elif t1 > 0:
                t = t1
            elif t2 > 0:
                t = t2
            else:
                return None  # Both intersections are behind the ray's origin
            
            intersection_point = ray.origin + t * ray.direction

            return t, self 
        
    def get_normal(self,intersection_point):
        return intersection_point - self.center

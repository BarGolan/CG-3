from helper_classes import *
import matplotlib.pyplot as plt


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            # TODO

            intersection_obj_params = ray.nearest_intersected_object(objects)
            if intersection_obj_params is not None:
                min_t, nearest_object = intersection_obj_params
                intersection_point = ray.origin + min_t * ray.direction
                normal = normalize(
                    nearest_object.get_normal(intersection_point))
                # hit_point += 1e-4 * normalize(normal)

                color = get_color(ambient, objects, normal, lights,
                                  intersection_point, nearest_object, camera, 0, max_depth)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def get_color(ambient, objects: list[Object3D], normal, lights: list[LightSource], intersection_point, nearest_object: Object3D, origin, level, max_depth):

    if level > max_depth:
        return
    material_emission = 0
    ambient = nearest_object.ambient * ambient
    total_light = ambient

    for light_source in lights:
        blocked = is_intersection_point_blocked(
            intersection_point, objects, light_source)
        if blocked:
            continue

        diffuse_light = get_diffused_light(
            nearest_object, normal, light_source, intersection_point)

        specular_light = get_specular_light(
            nearest_object, origin, intersection_point, light_source, normal)

        total_light += light_source.get_intensity(
            intersection_point) * (diffuse_light + specular_light)

    reflected_params = get_reflected_params(
        objects, intersection_point, origin=origin, normal=normal)
    if reflected_params:
        new_normal, new_intersection_point, new_nearest_obj = reflected_params
        print(nearest_object.reflection)
        total_light += nearest_object.reflection * \
            get_color(ambient, objects, new_normal, lights, new_intersection_point,
                      new_nearest_obj, intersection_point, level+1, max_depth)

    return total_light


def get_reflected_direction(ray_direction, normal):
    L = ray_direction
    N = normal
    reflected_direction = (L - 2 * (np.dot(N, L)) * N)

    return normalize(reflected_direction)


def is_intersection_point_blocked(
    intersection_point, objects: list[Object3D], light: LightSource
) -> bool:
    shadow_ray = Ray(intersection_point, light.position - intersection_point)
    intersection = shadow_ray.nearest_intersected_object(objects)
    return not intersection


def get_diffused_light(nearest_object: Object3D, normal, light_source: LightSource, intersection_point):
    return nearest_object.diffuse*(np.dot(normal, light_source.get_light_ray(intersection_point).direction))


def get_specular_light(nearest_object: Object3D, origin, intersection_point, light_source: LightSource, normal):
    V = normalize(origin - intersection_point)
    R = get_reflected_direction(-light_source.get_light_ray(
        intersection_point).direction, normal=normal)
    specular_light = (nearest_object.specular * np.dot(V, R)
                      ** nearest_object.shininess)

    return specular_light


def get_reflected_params(objects: list[Object3D], intersection_point, origin, normal):
    view_ray = Ray(origin, intersection_point-origin)
    reflected_ray = Ray(intersection_point,
                        get_reflected_direction(view_ray.direction, normal))
    intersection = reflected_ray.nearest_intersected_object(objects)
    if intersection:
        min_t, nearest_object = reflected_ray.nearest_intersected_object(
            objects)
        intersection_point = reflected_ray.origin + min_t * reflected_ray.direction
        new_normal = normalize(nearest_object.get_normal(intersection_point))
        return (new_normal, intersection_point, nearest_object)

    return None


def your_own_scene():
    camera = np.array([0, 0, 1])
    lights = []
    objects = []
    return camera, lights, objects

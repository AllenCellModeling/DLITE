import heapq
import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage as nimg
import argparse
import os
import collections

from PIL import Image
from scipy.spatial import distance
from skimage.filters import threshold_triangle
from skimage.morphology import skeletonize

"""
How the junction/edge extraction works:

First, we want to pick out all potential junctions. To do so, we will apply a morphological operation on the image, using
kernels with shapes of possible junctions. Then, we compute the dges from the "critical" points of each junction. We 
define points to be "critical" if they are within one pixel away from a junction. We then start traversing from the
critical points to other junctions if possible, preferring to traverse to the closest non-critical points (we should 
be reluctant to traverse towards other critical points, since they can potentially be the start of their own edge). In 
the case that a junction is found or the edge extends pasts the image boundary, we consider that a valid edge and store
it. If neither case occurs, the edge is invalid, as well as the junction. Repeat the process for all junctions and 
critical points to eventually find all true junctions.
"""


class EdgeExtractor:
    """
        Parameters:

        scan_distance: float, optional, default: 0.33
            Value ranging from 0.0 to 1.0. Higher values are used when we want the edge extraction to be more lenient
            when detecting edges in an image (i.e. when do we say a disconnected line in image is an edge or not).

        boundary_treatment: string, optional, default: "loose"
            There are two modes:
            "strict" considers lines extending past the boundary to not be edges.
            "loose" does the opposite.
    """

    def __init__(self, scan_distance=None, boundary_treatment=None):
        self._scan_distance = 0.66 if not scan_distance else min(max(scan_distance, 0.0), 1.0)
        self._boundary_treatment = "strict" if boundary_treatment == "strict" else "loose"
        self._gaussian_blur_sigma = 5

    """
        Parameters:
        
        image: string or numpy 2d matrix
            Path to a gray-scale image, or a 2D matrix representing a gray-scale image. This is the file/image which
            edges will be extracted out of.
        
        output_file: string, optional, default: None
            Path to where the extracted edge coordinates are written to.
        
        Returns a tuple (res, edges), where res is the processed image, and edges is the list of edges extracted.
    """

    def run(self, image, output_file=None, preprocess=False):
        if isinstance(image, str):
            image = mpimg.imread(image)

        img = image.copy()

        if preprocess:
            nimg.gaussian_filter(img, self._gaussian_blur_sigma, output=img)
            img[:] = (img > threshold_triangle(img))
            img[:] = skeletonize(img > 0)
            img *= 255

        return self._extract_edges(img, output_file)

    """
        Parameters:
        
        images: list of strings or list of numpy 2d matrices
            These are the files/images which edges will be extracted out of.
        
        output_file_prefix: string, optional, default: None
            Path prefix of the files where extracted edge coordinates will be written to. 
            The file will be named "<output_file_prefix>_<i>"for the image at index i of the images array. 
        
        Returns a list of 2-tuples where the first element is the processed image, and the second element is the list of extracted edges.
    """

    def run_series(self, images, output_file=None, preprocess=False):
        res = []

        file_name, ext = (None, None) if output_file is None else os.path.splitext(output_file)
        for i in range(len(images)):
            output_file_i = "{prefix}_{idx}{ext}".format(prefix=file_name, idx=i, ext=ext) if output_file else None
            res.append(self.run(images[i], output_file=output_file_i, preprocess=preprocess))
        return res

    """
        Method that does the edge extraction. 
        Called by the run method, and will write the edges out to a file if defined.
        
        Returns a tuple (res, edges), where res is the processed image, and edges is the list of edges extracted.
    """

    def _extract_edges(self, img, output_file=None):
        potential_junctions = self._generate_potential_junctions(img)
        res, valid_junctions, edges = self._compute_valid_junctions(img, potential_junctions)

        while len(potential_junctions) != len(valid_junctions):
            potential_junctions = valid_junctions
            res, valid_junctions, edges = self._compute_valid_junctions(res, potential_junctions)

        # write out the coordinates for each edge to the defined output file
        if output_file:
            with open(output_file, "w") as f:
                edge_count = 0
                for edge in edges:
                    f.write("EDGE_{}:\n".format(edge_count))
                    for x, y in edge:
                        formatted_pt = "{} {}".format(x, y)
                        f.write(formatted_pt + "\n")
                    edge_count += 1

        return res, edges

    """
        Returns a new graph image, with a set of valid junctions and edges based off potential junctions.
        Needs to be run again if we found junctions that are invalid.
    """
    def _compute_valid_junctions(self, img, potential_junctions):

        # How many pixels ahead do we look towards before we give up and say that this is not an edge.
        D = int(self._avg_nn_dist(potential_junctions) * self._scan_distance)

        # Critical points are points in edges that have junctions as one of their immediate neighbors.
        NON_CRITICAL_POINT = 0
        CRITICAL_POINT = 0.1

        edges = []

        potential_junctions = self._merge_immediate_junctions(potential_junctions)
        critical_points = collections.defaultdict(list)

        for j in list(potential_junctions):
            neighbors = get_neighbors(j[0], j[1])
            for x, y in neighbors:
                if within_bounds(img, x, y) and img[y][x] > 0:
                    critical_points[j].append((x,y))

        # don't recompute edges
        reserved_points = set()
        for j in list(potential_junctions):

            edge_starts = []

            # How many edges branch out of this junction?
            for x, y in critical_points[j]:
                # We want to start computing edges closest to the junction first
                # This helps solve T junctions since our algorithm can't traverse points twice
                d = distance.euclidean([j[0], j[1]], [x, y])

                # Store neighbor to traverse
                heapq.heappush(edge_starts, (d, x, y))

            # Compute edges that branch out of a junction
            while edge_starts:

                start_point = heapq.heappop(edge_starts)
                start_point = (start_point[1], start_point[2])

                # This starting point turned out to be a part of prior edge computation (T junction?)
                if start_point in reserved_points:
                    continue

                # Save the points we compute over the edge
                edge_points = {j}
                path = [j, start_point]

                # Loop will end in one of three cases
                # 1. edge traversal leads to a junction, we save all the edge points
                # 2. edge traversal ends without finding a junction, we remove all the edge points
                # 3. edge traversal extends past image boundary; boundary treatment dictates what to do with edge points
                while path:
                    edge_x, edge_y = path[-1]
                    edge_points = set(path)

                    # We've reached another junction! The points in this edge have been fully computed
                    if (edge_x, edge_y) in potential_junctions:
                        break

                    neighbors = get_neighbors(edge_x, edge_y)
                    candidate_points = []

                    # Traverse any immediate neighbor we haven't visited yet
                    has_immediate_neighbor = False

                    # If the current point extends past the image bounds, we consider it to have a neighbor
                    # depending on how we treat the boundary
                    if self._boundary_treatment == "loose":
                        has_immediate_neighbor = not within_bounds(img, edge_x, edge_y, D)

                    for x, y in neighbors:
                        if (x, y) not in reserved_points.union(edge_points) and (x, y) != j:
                            edge_within_image_bounds = within_bounds(img, x, y)
                            if edge_within_image_bounds and img[y][x] > 0:
                                d = distance.euclidean([edge_x, edge_y], [x, y])

                                # make sure we prefer non-critical points if there are two equidistant neighbors
                                critical = CRITICAL_POINT if (x, y) in critical_points[j] else NON_CRITICAL_POINT

                                heapq.heappush(candidate_points, (d + critical, x, y))
                                has_immediate_neighbor = True

                    # If there's no immediate neighbor, extend the search distance
                    if not has_immediate_neighbor:

                        n = D//2

                        if len(path) >= n:

                            x, y = path[-n]

                            # We compute all points within a triangle based off the angle between
                            # a past point in the path and the current point.
                            dx, dy = (edge_x - x), (edge_y - y)

                            scaled_dx = dx/max(abs(dx),abs(dy))
                            scaled_dy = dy/max(abs(dx),abs(dy))

                            # Calculate the vertices of the triangle
                            v1 = (int(edge_x - scaled_dx), int(edge_y - scaled_dy))

                            # D * slope + D/2 * perpendicular line
                            v2 = (int(v1[0] + D * scaled_dx - D//2 * scaled_dy), int(v1[1] + D * scaled_dy + D//2 * scaled_dx))

                            # D * slope - D/2 * perpendicular line
                            v3 = (int(v1[0] + D * scaled_dx + D//2 * scaled_dy), int(v1[1] + D * scaled_dy - D//2 * scaled_dx))

                            triangle_points = get_points_in_triangle(v1, v2, v3)

                            for x, y in triangle_points:

                                edge_past_image_bounds = not within_bounds(img, x, y)
                                if edge_past_image_bounds or img[y][x] == 0 or (x, y) in reserved_points.union(edge_points):
                                    continue

                                d = distance.euclidean([edge_x, edge_y], [x, y])
                                critical = CRITICAL_POINT if (x, y) in critical_points[j] else NON_CRITICAL_POINT

                                heapq.heappush(candidate_points, (d + critical, x, y))

                    if not has_immediate_neighbor and not candidate_points:
                        path = None
                    # Traverse to the best (closest) candidate point.
                    elif candidate_points:
                        _, x, y = heapq.heappop(candidate_points)
                        path.append((x,y))
                    else:
                        break

                # TODO: have nodes be first and last element in list
                if path:
                    edges.append(path)

                    # Junctions can be part of multiple edges, don't add them to the reserved point set
                    reserved_points = reserved_points.union(path).difference(potential_junctions)

        # Compute which junctions are valid
        true_junctions = []
        for j in list(potential_junctions):

            true_junction = True
            for point in critical_points[j]:
                if point not in reserved_points:
                    true_junction = False
                    break

            if true_junction:
                true_junctions.append(j)

        computed_edges = [item for sublist in edges for item in sublist]
        x = [t[0] for t in computed_edges]
        y = [t[1] for t in computed_edges]
        z = [255 for _ in computed_edges]

        res = np.zeros_like(img)
        res[y, x] = z

        return res, true_junctions, edges

    """
        Returns a list of (x,y) coordinates indicating potential junctions in an image.
        Uses a group of junction kernels to pattern match within the given image.
    """

    def _generate_potential_junctions(self, img):
        junction_kernels = self._generate_junction_kernels()

        junction_candidates = None
        for i, jk in enumerate(junction_kernels):
            junctions = nimg.binary_erosion(input=img, structure=jk)

            if junction_candidates is None:
                junction_candidates = junctions
            else:
                junction_candidates = np.logical_or(junction_candidates, junctions)

        junction_candidates = [(j[1], j[0]) for j in np.argwhere(junction_candidates).tolist()]

        return junction_candidates

    """
        Returns a list of kernels used for the morphological operation to detect junctions.    
    """

    def _generate_junction_kernels(self):
        base_junction_kernels = [
            [[1, 0, 1], [0, 1, 0], [0, 1, 0]],
            [[1, 0, 1], [0, 1, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 0, 0]]
        ]

        all_junction_kernels = []

        for k in base_junction_kernels:
            kernel = k
            for i in range(4):
                kernel = np.rot90(kernel)
                all_junction_kernels.append(kernel)

        return all_junction_kernels

    """
        Returns the average of the distances of every point's nearest neighbor.
        Used by the edge extractor when determining how far it should scan for pixels when computing an edge.
    """

    def _avg_nn_dist(self, pts):

        if len(pts) < 2:
            return 0

        # calculate distance to nearest neighboring junction and then average the distances
        avg_d = 0

        for i in range(len(pts)):

            min_d = float("inf")
            for j in range(len(pts)):
                if i == j:
                    continue

                d = distance.euclidean(pts[i], pts[j])
                min_d = min(min_d, d)

            avg_d += min_d

        return avg_d / len(pts)

    def _merge_immediate_junctions(self, junctions):
        to_remove = set()
        for j in junctions:
            if j in to_remove:
                continue

            neighbors = get_neighbors(j[0], j[1])
            for n in neighbors:
                if n in junctions:
                    to_remove.add(n)

        return list(set(junctions).difference(to_remove))

"""
    Triangle Utils
"""


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def is_point_in_triangle(pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def get_points_in_triangle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    min_x, max_x = min(x1, x2, x3), max(x1, x2, x3)
    min_y, max_y = min(y1, y2, y3), max(y1, y2, y3)

    triangle_points = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if is_point_in_triangle((x, y), p1, p2, p3):
                triangle_points.append((x, y))

    return triangle_points


"""
    Misc. Utils
"""


def within_bounds(img, x, y, n=0):
    y_bound = n <= y < len(img) - n
    x_bound = n <= x < len(img[0]) - n
    return x_bound and y_bound


def get_neighbors(x, y):
    return [(c, r) for c in range(x - 1, x + 2) for r in range(y - 1, y + 2) if r != y or c != x]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="input to the extractor, which can be a path to an image file or directory of image files")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="name of the output file where edge coordinates will be saved (will append a suffixes if the input is a directory)")
    parser.add_argument("-m", "--images", action="store_true", help="creates and saves images from the extracted edges using the same filename as the output")
    parser.add_argument("-s", "--scan_range", type=float, default=0.66, help="value from 0.0 to 1.0 determining how far the algorithm should scan for pixels a part of an edge, 1.0 being up to the average distance between nearest nodes")
    parser.add_argument("-b", "--boundary_mode", type=str, choices=["strict", "loose"], default="loose", help="modes deciding whether to interpret lines extending past an image boundary as edges (loose is yes, strict is no)")
    parser.add_argument("-p", "--preprocess", action="store_true", help="include if the input is not binarized and skeletonized")
    args = parser.parse_args()

    extractor = EdgeExtractor(args.scan_range, args.boundary_mode)
    output_filename, output_ext = os.path.splitext(args.output)

    if os.path.isdir(args.input):
        valid_img_extensions = ['.tif', '.png', '.jpg']
        dirpath, dirs, files = next(os.walk(args.input))

        # full paths of files in directory if file has a supported image extension
        full_paths = [os.path.join(dirpath,f) for f in files if os.path.splitext(f)[1] in valid_img_extensions]

        for i in range(len(full_paths)):
            input_file = str(full_paths[i])
            output_file_i = "{prefix}_{idx}{ext}".format(prefix=output_filename, idx=i, ext=output_ext)
            res = extractor.run(input_file, output_file=output_file_i, preprocess=args.preprocess)

            if args.images:
                img = Image.fromarray(res[0])
                img.save("{prefix}_{idx}.png".format(prefix=output_filename,idx=i))

    elif os.path.isfile(args.input):
        res = extractor.run(args.input, args.output, preprocess=args.preprocess)

        if args.images:
            img = Image.fromarray(res[0])
            img.save("{}.png".format(output_filename))

import heapq
import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage as nimg

from scipy.spatial import distance
from skimage.filters import threshold_yen
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
        self._scan_distance = 0.33 if not scan_distance else min(max(scan_distance, 0.0), 1.0)
        self._boundary_treatment = "strict" if boundary_treatment == "strict" else "loose"
        self._gaussian_blur_sigma = 3

    """
        Parameters:
        
        image: string or numpy 2d matrix
            Path to a gray-scale image, or a 2D matrix representing a gray-scale image. This is the file/image which
            edges will be extracted out of.
        
        output_file: string, optional, default: None
            Path to where the extracted edge coordinates are written to.
        
        Returns a tuple (res, edges), where res is the processed image, and edges is the list of edges extracted.
    """
    def run(self, image, output_file=None):
        if isinstance(image, str):
            image = mpimg.imread(image)

        img = image.copy()
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
    def run_series(self, images, output_file_prefix=None):
        res = []
        for i in range(len(images)):
            output_file = "{prefix}_{idx}".format(prefix=output_file_prefix, idx=i) if output_file_prefix else None
            res.append(self.run(images[i], output_file=output_file))
        return res

    """
        Method that does the edge extraction. 
        Called by the run method, and will write the edges out to a file if defined.
        
        Returns a tuple (res, edges), where res is the processed image, and edges is the list of edges extracted.
    """
    def _extract_edges(self, img, output_file=None):
        nimg.gaussian_filter(img, self._gaussian_blur_sigma, output=img)
        img[:] = (img > threshold_yen(img))
        img[:] = skeletonize(img > 0)
        img *= 255

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
        CRITICAL_POINT = 1

        true_junctions = []
        edges = []

        # don't recompute edges
        reserved_points = set()

        for j in list(potential_junctions):

            # Start off assuming the potential junction is valid
            valid_junction = True

            edge_starts = []
            neighbors = get_neighbors(j[0], j[1])

            # How many edges branch out of this junction?
            for x, y in neighbors:
                if img[y][x] > 0 and (x, y) not in reserved_points:
                    # We want to start computing edges closest to the junction first
                    # This helps solve T junctions since our algorithm can't traverse points twice
                    d = distance.euclidean([j[0], j[1]], [x, y])

                    # Use this when calculating triangles
                    dx, dy = x - j[0], y - j[1]

                    # Store neighbor to traverse, and the direction of the neighbor
                    heapq.heappush(edge_starts, (d, x, y, dx, dy))

            critical_points = [(e[1], e[2]) for e in edge_starts]

            # Compute edges that branch out of a junction
            while edge_starts:

                start_point = heapq.heappop(edge_starts)

                # This starting point turned out to be a part of prior edge computation (T junction?)
                if (start_point[1], start_point[2]) in reserved_points:
                    continue

                # Save the points we compute over the edge
                edge_points = set()

                path = [start_point]
                while path:
                    dist, edge_x, edge_y, diff_x, diff_y = path.pop()

                    edge_points.add((edge_x, edge_y))

                    # We've reached another junction! The points in this edge have been fully computed
                    # Junctions can be part of multiple edges, don't add them to the reserved point set
                    if (edge_x, edge_y) in potential_junctions:
                        break

                    reserved_points.add((edge_x, edge_y))

                    neighbors = get_neighbors(edge_x, edge_y)
                    candidate_points = []

                    # Traverse any immediate neighbor we haven't visited yet
                    has_immediate_neighbor = False

                    # If the current point extends past the image bounds, we consider it to have a neighbor
                    # depending on how we treat the boundary
                    if self._boundary_treatment == "loose":
                        has_immediate_neighbor = not within_bounds(img, edge_x, edge_y, D)

                    for x, y in neighbors:
                        if (x, y) not in reserved_points and (x, y) != j:
                            edge_within_image_bounds = within_bounds(img, x, y)
                            if edge_within_image_bounds and img[y][x] > 0:
                                d = distance.euclidean([edge_x, edge_y], [x, y])
                                dx, dy = x - edge_x, y - edge_y
                                critical = CRITICAL_POINT if (x, y) in critical_points else NON_CRITICAL_POINT

                                heapq.heappush(candidate_points, (d + critical, x, y, dx, dy))
                                has_immediate_neighbor = True


                    # If there's no immediate neighbor, extend the search distance
                    if not has_immediate_neighbor:

                        # We compute all points within a triangle based off the angle between
                        # the prior point and the current point.

                        # Calculate the vertices of the triangle
                        v1 = (edge_x - diff_x, edge_y - diff_y)

                        # D * slope + D * perpendicular line
                        v2 = (v1[0] + D*diff_x - D*diff_y, v1[1] + D*diff_y + D*diff_x)

                        # D * slope - D * perpendicular line
                        v3 = (v1[0] + D*diff_x + D*diff_y, v1[1] + D*diff_y - D*diff_x)

                        triangle_points = get_points_in_triangle(v1, v2, v3)

                        for x, y in triangle_points:

                            edge_past_image_bounds = not within_bounds(img, x, y)
                            if edge_past_image_bounds or img[y][x] == 0 or (x, y) in reserved_points:
                                continue

                            d = distance.euclidean([edge_x, edge_y], [x, y])
                            critical = CRITICAL_POINT if (x, y) in critical_points else NON_CRITICAL_POINT

                            dx, dy = x - edge_x, y - edge_y
                            scale_back = max(abs(dx), abs(dy))

                            heapq.heappush(candidate_points, (d + critical, x, y, dx // scale_back, dy // scale_back))

                    # Traverse to the best (closest) candidate point.
                    if candidate_points:
                        path.append(heapq.heappop(candidate_points))

                    if not has_immediate_neighbor and not candidate_points:
                        edge_points = None
                        valid_junction = False

                if edge_points:
                    edges.append(sorted(list(edge_points)))

            if valid_junction:
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

from PIL import Image

a = EdgeExtractor()
img = a.run("../Notebooks/Data/ZO-1_data/Time-series_2/20170123_I01_003.czi - 20170123_I01_0030004.tif")[0]
img = Image.fromarray(img)
img.save("test_please.png")
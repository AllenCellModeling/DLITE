       
## METHOD 1

        nodes = self.nodes
        edges = self.edges

        # solve y = Ax
        # initialize A
        A = np.zeros((len(edges), 1))
        y = np.zeros(len(edges))

        for node in nodes:
                # node.edges should give the same edge as the edge corresponding to node.tension_vectors since they are added together
                # only want to add indices of edges that are part of colony edge list
                indices = np.array([edges.index(x) for x in node.edges if x in edges])
                # only want to consider horizontal vectors that are a part of the colony edge list 
                horizontal_vectors = np.array([x[0] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges])[np.newaxis]
                A = np.append(A, horizontal_vectors.T, axis=1)

                vertical_vectors = np.array([x[1] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges])[np.newaxis]
                A = np.append(A, vertical_vectors.T, axis=1)

        #tensions = np.linalg.solve(A, B) # doesnt work for homogenous system
        #tensions = solution(A)
        tensions= np.linalg.lstsq(A, y, rcond = None)[0]

# METHOD 2

        nodes = self.nodes
        edges = self.edges

        # solve AX = B
        # initialize A
        A = np.zeros((len(edges), len(edges)))

        j = 0

        for node in nodes:

            if j < len(edges):

                # node.edges should give the same edge as the edge corresponding to node.tension_vectors since they are added together
                # only want to add indices of edges that are part of the edges of the colony
                indices = np.array([edges.index(x) for x in node.edges if x in edges])
                # only want to consider horizontal vectors that are a part of the colony edge list 
                horizontal_vectors = np.array([x[0] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges])

                A[j, indices] = horizontal_vectors
                j +=1

        if j < len(edges):
            for node in nodes:
                if j < len(edges):
                    indices = np.array([edges.index(x) for x in node.edges if x in edges])
                    vertical_vectors = np.array([x[1] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges])
                    A[j,indices] = vertical_vectors
                    j += 1

        #tensions = np.linalg.solve(A, B) # doesnt work for homogenous system
        tensions = solution(A)
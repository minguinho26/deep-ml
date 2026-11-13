import numpy as np 

def get_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    # Your code here
    
    final_centroids = initial_centroids

    for n in range(max_iterations):
    
        new_centroids = []
        for i in range(len(final_centroids)):
            new_centroids.append([])
        
        for p in points:
            min_d = float("inf")
            target_c_idx = 0
            for c_idx, c in enumerate(final_centroids):
                d = get_distance(p, c)
                
                if min_d > d:
                    min_d = d 
                    target_c_idx = c_idx
            
            new_centroids[target_c_idx].append(p)
        
        # 새로운 중점 정하기 
        for i in range(len(final_centroids)):
            new_centroids[i] = tuple(np.array(new_centroids[i]).mean(axis=0).tolist())
        
        final_centroids = new_centroids

    return final_centroids
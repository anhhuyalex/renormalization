import hashlib
import numpy as np
import glob
from collections import defaultdict

def get_key(array):
    return hashlib.sha256(array.tostring()).hexdigest()

def normalize_counts(p_dict):
    normalizer = sum(p_dict.values())
    for config, count in p_dict.items():
        p_dict[config] = count / normalizer
    return p_dict



def mutual_info_fine_grained_5(data):
    """
    boundary is the 3x3 bordering strip and the 2 non-central spins bordering the border
    x *x* *x* | *x* *x* x
    x  x  *x* | *x*  x  x
    x *x* *x* | *x* *x* x
    """
    data = np.sign(data) # discretize data
    data = data.astype("float64")
    # print(f"data: {data}")
#     return

    n_samples, n_rows, n_columns = data.shape
    print(n_samples, n_rows, n_columns)

    # compute counts
    p_xy = defaultdict(int)
    p_x = defaultdict(int)
    p_y = defaultdict(int)
    n_samples = 300000 # for testing, remove later
    seen_boundaries = set()
    for _ in range(n_samples):

        sample = np.random.randint(data.shape[0])
#         print("sample", sample)
        row, col = np.random.randint(data.shape[-1], size = 2)
#         if sample % 100 == 0:
#             print(sample, p_xy)
        if (row > data.shape[-1] - 3) or (col > data.shape[-1] - 6):
            rectangle = np.pad(data[sample], 6)[row:(row + 3), col:(col+6)]
        else:
            rectangle = data[sample, row:(row + 3), col:(col+6)]
        boundary_x = np.hstack((rectangle[:, 2], rectangle[0, 1], rectangle[2, 1]))
        p_x[get_key(boundary_x)] += 1
        boundary_y = np.hstack((rectangle[:, 3], rectangle[0, 4], rectangle[2, 4]))
        p_y[get_key(boundary_y)] += 1
        joint_boundary = np.array([np.hstack((boundary_x, boundary_y))])
        p_xy[get_key(joint_boundary)] += 1
        seen_boundaries.add(joint_boundary.tostring())

#     mutual_info = sum_over_xy of p(x,y)*log(p)

    # Normalize counts
    p_xy = normalize_counts(p_xy)
    p_x = normalize_counts(p_x)
    p_y = normalize_counts(p_y)


#     # Generate all possible configurations of boundary conditions
#     n = 10
#     i = np.array(np.indices(n * (2,))).reshape(n, -1)
#     xy_keys = (i[:, np.argsort(i.sum(0)[::-1], kind='mergesort')].T[::-1].reshape(-1, n)*2-1)

    # mi = <log (p(x, y)/p(x)p(y))>
    mi = 0
    no_config = []

    for xy_string in seen_boundaries:
        xy = np.fromstring(xy_string)
        print(xy)
        joint_prob = p_xy[get_key(xy)]
        x = xy[0:5]
        x_prob = p_x[get_key(x)]
        y = xy[5:10]
        y_prob = p_y[get_key(y)]
#         print(f"boundary_x:{x}")
        if joint_prob != 0:
            mi += np.log2(joint_prob/(x_prob * y_prob)) * joint_prob
            no_config.append(xy)



    return p_xy, p_x, p_y, mi, no_config

# Start Ray.
import ray
ray.init()
#
@ray.remote
def count(chunk, mode = "individual"):
    p_xy = defaultdict(int)
    p_x = defaultdict(int)
    p_y = defaultdict(int)
    seen_boundaries = set()
    chunk = np.sign(chunk).astype("float64")
    for arr in chunk:
        if mode == "individual":
            data = np.load(arr)['arr_0'][None, :, :]
        elif mode == "batch":
            data = arr
        # data = np.sign(data).astype("float64")
        n_samples, n_rows, n_columns = data.shape

        # compute counts

        n_samples = 1000 # for testing, remove later

        for _ in range(n_samples):

            sample = 0 #np.random.randint(data.shape[0])
            row, col = np.random.randint(data.shape[-1], size = 2)
            if (row > data.shape[-1] - 3) or (col > data.shape[-1] - 6):
                rectangle = np.pad(data[sample], 6)[row:(row + 3), col:(col+6)]
            else:
                rectangle = data[sample, row:(row + 3), col:(col+6)]
            boundary_x = np.hstack((rectangle[:, 2], rectangle[0, 1], rectangle[2, 1]))
            p_x[get_key(boundary_x)] += 1
            boundary_y = np.hstack((rectangle[:, 3], rectangle[0, 4], rectangle[2, 4]))
            p_y[get_key(boundary_y)] += 1
            joint_boundary = np.array([np.hstack((boundary_x, boundary_y))])
            # print("joint_boundary", np.fromstring(joint_boundary.tostring()))
            p_xy[get_key(joint_boundary)] += 1
            seen_boundaries.add(joint_boundary.tostring())


    return p_xy, p_x, p_y, seen_boundaries

def computeMI(results):
    p_xy_totals = defaultdict(int)
    p_x_totals = defaultdict(int)
    p_y_totals = defaultdict(int)
    seen_boundaries_totals = set()
    for chunk in results:
        p_xy, p_x, p_y, seen_boundaries = chunk
        # print("seen_boundaries", [np.fromstring(x) for x in seen_boundaries])
        for key in p_xy:
            p_xy_totals[key] += p_xy[key]
        for key in p_x:
            p_x_totals[key] += p_x[key]
        for key in p_y:
            p_y_totals[key] += p_y[key]
        for key in seen_boundaries:
            seen_boundaries_totals.add(key)

    p_xy_totals = normalize_counts(p_xy_totals)
    p_x_totals =  normalize_counts(p_x_totals)
    p_y_totals =  normalize_counts(p_y_totals)


    mi = 0
    no_config = []

    for xy_string in seen_boundaries_totals:
        xy = np.fromstring(xy_string)
        # print(xy)
        joint_prob = p_xy_totals[get_key(xy)]
        x = xy[0:5]
        x_prob = p_x_totals[get_key(x)]
        y = xy[5:10]
        y_prob = p_y_totals[get_key(y)]
#         print(f"boundary_x:{x}")
        if joint_prob != 0:
            mi += np.log2(joint_prob/(x_prob * y_prob)) * joint_prob
            no_config.append(xy)



    return mi



if __name__ == "__main__":
    import sys
    if sys.argv[1] == "2187":
        files = glob.glob("../data_2187_1571810501/*")
        file_chunks = np.array_split(files[100:], 4)
        result_ids = []
        for chunk in range(4):
            result_ids.append(count.remote(file_chunks[chunk]))
        # Wait for the tasks to complete and retrieve the results.
        # With at least 4 cores, this will take 1 second.
        results = ray.get(result_ids)
        mi = computeMI(results)

    elif sys.argv[1] == "729":
        files = np.load(f"/Users/qanguyen/Downloads/ising_temp2.269_correlated729x729from2187x2187_batch2.npy")
        # file_chunks = np.array_split(files[100:], 4)
        # del files
        # print("file_chunks", file_chunks[0][:, None, :, :].shape)
        # print("file_chunks", file_chunks[0][0,:10,:10])
        # result_ids = []
        # for chunk in range(1):
        #     result_ids.append(count.remote((files[:, None, :, :]), mode = "batch")) # discretize to {-1, 1}
        p_xy, p_x, p_y, mi, no_config = mutual_info_fine_grained_5(files)
    else:
        files = np.load(f"/Users/qanguyen/Downloads/ising_temp2.269_correlated{sys.argv[1]}x{sys.argv[1]}from{int(sys.argv[1])*3}x{int(sys.argv[1])*3}.npy")
        file_chunks = np.array_split(files[100:], 4)
        del files
        # print("file_chunks", file_chunks[0][:, None, :, :].shape)
        print("file_chunks", file_chunks[0][0,:10,:10])
        result_ids = []
        for chunk in range(4):
            result_ids.append(count.remote((file_chunks[chunk][:, None, :, :]), mode = "batch")) # discretize to {-1, 1}

        results = ray.get(result_ids)
        mi = computeMI(results)
    # print("MI", (file_chunks[0]))
    print("MI", mi)
    with open("mi.txt", "a") as myfile:
        myfile.write(f"{sys.argv[1]}: {mi}\n")

from openselfsup.datasets.data_sources.mst_images import MSTImageList, MSTPairImageList
import pdb
root = '/mnt/fs4/chengxuz/hippocampus_change/mst_related/MST'


def main():
    #mst_data = MSTImageList(root=root)
    #mst_data = MSTImageList(root=root, which_set='Set 1')
    #mst_data = MSTImageList(root=root, oversample_len=1281167)
    mst_data = MSTPairImageList(root=root, oversample_len=1281167)
    pdb.set_trace()
    pass


if __name__ == '__main__':
    main()

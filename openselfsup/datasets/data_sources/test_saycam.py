from openselfsup.datasets.data_sources.saycam import \
        SAYCam, SAYCamTwoImageRandom, SAYCamCont
import pdb
root = '/data5/chengxuz/Dataset/infant_headcam/jpgs_extracted'


def two_image_test():
    list_file = '/mnt/fs1/Dataset/infant_headcam/meta_for_cont_curr.txt'
    #saycam_data = SAYCam(root=root, list_file=list_file, batch_size=256)
    saycam_data = SAYCamTwoImageRandom(
            root=root, list_file=list_file, batch_size=256)
    return saycam_data


def cont_test():
    list_file = '/mnt/fs1/Dataset/infant_headcam/meta_for_cont_bench.txt'
    num_frames_meta_file = '/mnt/fs1/Dataset/infant_headcam/num_frames_meta.txt'
    saycam_data = SAYCamCont(
            root=root, list_file=list_file, 
            num_frames_meta_file=num_frames_meta_file)
    return saycam_data


def main():
    #saycam_data = two_image_test()
    saycam_data = cont_test()
    pdb.set_trace()
    pass


if __name__ == '__main__':
    main()

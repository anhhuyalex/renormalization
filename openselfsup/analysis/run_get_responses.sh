for grating_folder in /mnt/fs4/chengxuz/rodent_dev/HMAX/gratings/0*
do
    for which_model in simclr_alxnt_ctl64 simclr_alxnt_sy_two_img_ctl64 simclr_alxnt_sy_two_img_rd_ctl64
    do
        for resize_size in 64 96 128
        do
            CUDA_VISIBLE_DEVICES=$1 python openselfsup/analysis/get_responses_for_gratings.py \
                --which_model ${which_model} --grating_folder ${grating_folder} \
                --resize_size ${resize_size}
        done
    done
done

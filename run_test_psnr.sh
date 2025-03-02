python3 test_video.py \
    --i_frame_model_name IntraNoAR \
    --i_frame_model_path ./Imodel/ckpt_q3.pth.tar \
                         ./Imodel/ckpt_q4.pth.tar \
                         ./Imodel/ckpt_q5.pth.tar \
                         ./Imodel/ckpt_q6.pth.tar \
    --model_path ./Pmodel/ckpt_q1.pth \
                 ./Pmodel/ckpt_q2.pth \
                 ./Pmodel/ckpt_q3.pth \
                 ./Pmodel/ckpt_q4.pth \
    --test_config recommended_test_full_results_IP32.json \
    --cuda 1 \
    -w 1 \
    --output_path result.json \
